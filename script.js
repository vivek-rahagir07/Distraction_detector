/* CORE_SENTINEL */
class FocusSentinel {
    constructor() {
        // UI_BINDINGS
        this.videoElement = document.getElementById('input_video');
        this.canvasElement = document.getElementById('output_canvas');
        this.canvasCtx = this.canvasElement.getContext('2d');
        this.statusBadge = document.getElementById('statusBadge');
        this.alertOverlay = document.getElementById('alertOverlay');
        this.setupOverlay = document.getElementById('setupOverlay');
        this.videoFrame = document.getElementById('videoFrame');
        this.requestBtn = document.getElementById('requestPermissionBtn');
        this.stopBtn = document.getElementById('stopBtn');

        // METRICS_UI
        this.logContainer = document.getElementById('logContainer');
        this.attentionScoreEl = document.getElementById('attentionScore');
        this.timerEl = document.getElementById('timer');
        this.violationCountEl = document.getElementById('violationCount');
        this.focusStateEl = document.getElementById('focusState');

        // POMO_UI
        this.pomoTimerEl = document.getElementById('pomoTimer');
        this.pomoStartBtn = document.getElementById('pomoStart');
        this.pomoResetBtn = document.getElementById('pomoReset');
        this.pomoLabelEl = document.getElementById('pomoLabel');

        // STATE_FLAGS
        this.isActive = false;
        this.violations = 0;
        this.startTime = null;
        this.timerInterval = null;
        this.framesActive = 0;
        this.framesFocused = 0;
        this.lastViolationTime = 0;

        // POMO_STATE
        this.pomoTimeLeft = 25 * 60;
        this.pomoInterval = null;
        this.isPomoRunning = false;
        this.pomoMode = 'focus';

        // ANALYTICS_DATA
        this.focusHistory = [];
        this.chart = null;

        // AUDIO_CTX
        this.audioCtx = null;

        // MEDIAPIPE_INIT
        this.faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        this.faceMesh.setOptions({
            maxNumFaces: 2,
            refineLandmarks: true,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7
        });

        this.initObjectDetector();

        this.camera = new Camera(this.videoElement, {
            onFrame: async () => {
                if (this.isActive) {
                    await this.faceMesh.send({ image: this.videoElement });
                    if (this.objectDetector) {
                        const results = await this.objectDetector.detect(this.videoElement);
                        this.onObjectResults(results);
                    }
                }
            },
            width: 640,
            height: 480
        });

        this.init();
    }

    async initObjectDetector() {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );
        this.objectDetector = await ObjectDetector.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite`,
                delegate: "GPU"
            },
            scoreThreshold: 0.5,
            runningMode: "VIDEO"
        });
    }

    onObjectResults(results) {
        if (!results.detections) return;

        const forbidden = ['cell phone', 'laptop', 'tablet', 'book'];
        for (const detection of results.detections) {
            const category = detection.categories[0].categoryName;
            if (forbidden.includes(category)) {
                this.handleViolation(`UNAUTHORIZED_DEVICE: ${category.toUpperCase()}`);
            }
        }
    }

    init() {
        this.faceMesh.onResults((results) => this.onResults(results));
        this.requestBtn.addEventListener('click', () => this.startSystem());
        this.stopBtn.addEventListener('click', () => location.reload());

        this.pomoStartBtn.addEventListener('click', () => this.togglePomo());
        this.pomoResetBtn.addEventListener('click', () => this.resetPomo());

        document.getElementById('downloadLog').addEventListener('click', () => this.downloadSessionData());

        this.initChart();
        console.log("SYS_INIT_OK");
    }

    initChart() {
        const ctx = document.getElementById('stabilityChart').getContext('2d');
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: Array(20).fill(''),
                datasets: [{
                    label: 'FOCUS_STABILITY',
                    data: Array(20).fill(0),
                    borderColor: '#00f3ff',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: true,
                    backgroundColor: 'rgba(0, 243, 255, 0.05)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: { display: false },
                    y: {
                        display: false,
                        min: 0,
                        max: 100
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    }

    async startSystem() {
        const btnText = this.requestBtn.querySelector('span');
        btnText.innerText = "AUTHENTICATING...";
        this.requestBtn.disabled = true;

        try {
            await this.camera.start();
            await this.startAudioAnalysis();
            this.isActive = true;
            this.setupOverlay.classList.add('hidden');
            this.stopBtn.classList.remove('hidden');
            this.updateStatus("SECURED", "border-red-500 text-red-500 bg-red-500/10", "bg-red-500");
            this.startTimer();

            this.canvasElement.width = this.videoElement.videoWidth || 640;
            this.canvasElement.height = this.videoElement.videoHeight || 480;

            this.addLog("PROTOCOLS_ENGAGED", "success");
            this.startPomo(); // Start monitoring immediately
        } catch (err) {
            console.error(err);
            const errorMsg = document.getElementById('errorMessage');
            errorMsg.innerText = "SECURITY_ERR: ACCESS_DENIED";
            errorMsg.classList.remove('hidden');
            btnText.innerText = "RE-INITIALIZE";
            this.requestBtn.disabled = false;
        }
    }

    async startAudioAnalysis() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.initAudio();
            const source = this.audioCtx.createMediaStreamSource(stream);
            const analyser = this.audioCtx.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);

            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            this.audioInterval = setInterval(() => {
                analyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((p, c) => p + c, 0) / dataArray.length;
                if (average > 40) { // Threshold for talking/noise
                    this.handleViolation("VOCAL_DETECTION");
                }
            }, 1000);
        } catch (err) {
            console.warn("Audio monitoring disabled: ", err);
        }
    }

    onResults(results) {
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        this.canvasCtx.drawImage(results.image, 0, 0, this.canvasElement.width, this.canvasElement.height);

        this.framesActive++;

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0];

            drawConnectors(this.canvasCtx, landmarks, FACEMESH_TESSELATION, { color: '#00f3ff15', lineWidth: 0.5 });
            drawConnectors(this.canvasCtx, landmarks, FACEMESH_RIGHT_EYE, { color: '#ff00c1', lineWidth: 1.5 });
            drawConnectors(this.canvasCtx, landmarks, FACEMESH_LEFT_EYE, { color: '#ff00c1', lineWidth: 1.5 });
            drawConnectors(this.canvasCtx, landmarks, FACEMESH_LIPS, { color: '#00f3ff', lineWidth: 1 });

            const isLookingAway = this.checkLookingAway(landmarks);
            const multiplePeople = results.multiFaceLandmarks.length > 1;

            if (isLookingAway || multiplePeople) {
                const reason = isLookingAway ? "GAZE_DEV" : "MULTI_ENT";
                this.handleViolation(reason);
                this.updateUIForDistraction(true);
            } else {
                this.framesFocused++;
                this.updateUIForDistraction(false);
            }
        } else {
            this.handleViolation("SIG_LOST");
            this.updateUIForDistraction(true, "LOST");
        }

        this.updateMetrics();
        this.canvasCtx.restore();
    }

    checkLookingAway(landmarks) {
        const leftEye = landmarks[33];
        const rightEye = landmarks[263];
        const nose = landmarks[1];
        const eyeDist = rightEye.x - leftEye.x;
        const noseRelativeX = (nose.x - leftEye.x) / eyeDist;
        const forehead = landmarks[10];
        const chin = landmarks[152];
        const faceHeight = chin.y - forehead.y;
        const noseRelativeY = (nose.y - forehead.y) / faceHeight;
        return (noseRelativeX < 0.35 || noseRelativeX > 0.65 || noseRelativeY < 0.35 || noseRelativeY > 0.75);
    }

    initAudio() {
        if (this.audioCtx) return;
        this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }

    playTone(freq, type, duration, vol) {
        if (!this.audioCtx) return;
        const osc = this.audioCtx.createOscillator();
        const gain = this.audioCtx.createGain();
        osc.type = type;
        osc.frequency.setValueAtTime(freq, this.audioCtx.currentTime);
        gain.gain.setValueAtTime(vol, this.audioCtx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.0001, this.audioCtx.currentTime + duration);
        osc.connect(gain);
        gain.connect(this.audioCtx.destination);
        osc.start();
        osc.stop(this.audioCtx.currentTime + duration);
    }

    playAlert() {
        this.initAudio();
        this.playTone(150, 'sawtooth', 0.5, 0.1);
        setTimeout(() => this.playTone(100, 'sawtooth', 0.5, 0.1), 100);
    }

    playSuccess() {
        this.initAudio();
        this.playTone(880, 'sine', 0.8, 0.1);
        setTimeout(() => this.playTone(1100, 'sine', 0.8, 0.1), 150);
    }

    handleViolation(type) {
        if (this.pomoMode === 'break' || !this.isPomoRunning) return;
        const now = Date.now();
        if (now - this.lastViolationTime > 4000) {
            this.violations++;
            this.violationCountEl.innerText = this.violations;
            this.addLog(type, "alert");
            this.lastViolationTime = now;
            this.videoFrame.classList.add('alert-shake');
            setTimeout(() => this.videoFrame.classList.remove('alert-shake'), 500);
            this.playAlert();
        }
    }

    updateUIForDistraction(isDistracted, customState = null) {
        if (isDistracted) {
            this.updateStatus(customState || "SECURITY_ALERT", "border-red-500 text-red-500 bg-red-500/10", "bg-red-500");
            this.alertOverlay.style.opacity = "1";
            this.videoFrame.classList.add('neon-border-red');
            this.videoFrame.classList.remove('neon-border-green', 'neon-border-cyan');
            this.focusStateEl.innerText = customState || "INTEGRITY_COMPROMISED";
            this.focusStateEl.className = "text-xl font-bold text-red-500 uppercase glow-text-red";
        } else {
            this.updateStatus("SECURED", "border-red-500 text-red-400 bg-red-500/10", "bg-red-500");
            this.alertOverlay.style.opacity = "0";
            this.videoFrame.classList.add('neon-border-green');
            this.videoFrame.classList.remove('neon-border-red', 'neon-border-cyan');
            this.focusStateEl.innerText = "COMPLIANT";
            this.focusStateEl.className = "text-xl font-bold text-green-400 uppercase";
        }
    }

    addLog(message, type = "info") {
        if (this.logContainer.innerText.includes("Awaiting pulse signal")) {
            this.logContainer.innerHTML = "";
        }
        const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        const logEntry = document.createElement('div');
        const borderClass = type === 'alert' ? 'border-red-500/30' : (type === 'success' ? 'border-green-500/30' : 'border-cyan-500/30');
        const labelClass = type === 'alert' ? 'text-red-500' : (type === 'success' ? 'text-green-500' : 'text-cyan-500');
        logEntry.className = `p-4 glass-panel ${borderClass} rounded-xl fade-in mb-3`;
        logEntry.innerHTML = `
            <div class="flex justify-between items-center mb-1">
                <span class="text-[10px] font-black ${labelClass} uppercase tracking-widest">${type}</span>
                <span class="text-[10px] text-slate-500 font-mono">${timeStr}</span>
            </div>
            <p class="text-sm text-slate-300 font-bold">${message}</p>
        `;
        this.logContainer.prepend(logEntry);
    }

    updateMetrics() {
        const ratio = this.framesActive > 0 ? (this.framesFocused / this.framesActive) * 100 : 0;
        this.attentionScoreEl.innerText = Math.round(ratio) + "%";
        if (this.framesActive % 30 === 0) {
            this.chart.data.datasets[0].data.shift();
            this.chart.data.datasets[0].data.push(ratio);
            this.chart.update('none');
            const consistency = Math.round(ratio);
            document.getElementById('consistencyScore').innerText = consistency + "%";
            document.getElementById('consistencyBar').style.width = consistency + "%";
        }
        if (ratio > 80) this.attentionScoreEl.style.color = 'var(--neon-green)';
        else if (ratio > 50) this.attentionScoreEl.style.color = 'var(--neon-cyan)';
        else this.attentionScoreEl.style.color = 'var(--neon-red)';
    }

    startTimer() {
        this.startTime = Date.now();
        this.timerInterval = setInterval(() => {
            const diff = Date.now() - this.startTime;
            const min = Math.floor(diff / 60000).toString().padStart(2, '0');
            const sec = Math.floor((diff % 60000) / 1000).toString().padStart(2, '0');
            this.timerEl.innerText = `${min}:${sec}`;
        }, 1000);
    }

    togglePomo() {
        if (this.isPomoRunning) { this.pausePomo(); } else { this.startPomo(); }
    }

    startPomo() {
        this.isPomoRunning = true;
        this.pomoStartBtn.innerText = "PAUSE";
        this.pomoInterval = setInterval(() => {
            if (this.pomoTimeLeft > 0) {
                this.pomoTimeLeft--;
                this.updatePomoUI();
            } else { this.switchPomoMode(); }
        }, 1000);
        this.addLog(`PHASE_${this.pomoMode.toUpperCase()}_START`, "info");
    }

    pausePomo() {
        this.isPomoRunning = false;
        this.pomoStartBtn.innerText = "RESUME";
        clearInterval(this.pomoInterval);
    }

    resetPomo() {
        this.pausePomo();
        this.pomoTimeLeft = this.pomoMode === 'focus' ? 25 * 60 : 5 * 60;
        this.pomoStartBtn.innerText = "START";
        this.updatePomoUI();
    }

    switchPomoMode() {
        this.pomoMode = this.pomoMode === 'focus' ? 'break' : 'focus';
        this.pomoTimeLeft = this.pomoMode === 'focus' ? 25 * 60 : 5 * 60;
        this.pomoLabelEl.innerText = this.pomoMode.toUpperCase();
        this.pomoLabelEl.className = `text-[10px] px-2 py-0.5 ${this.pomoMode === 'focus' ? 'bg-pink-500/20 text-pink-400' : 'bg-green-500/20 text-green-400'} rounded-full font-bold`;
        this.addLog(`PHASE_SWITCH: ${this.pomoMode.toUpperCase()}`, "success");
        this.playSuccess();
        this.updatePomoUI();
    }

    updatePomoUI() {
        const min = Math.floor(this.pomoTimeLeft / 60).toString().padStart(2, '0');
        const sec = (this.pomoTimeLeft % 60).toString().padStart(2, '0');
        this.pomoTimerEl.innerText = `${min}:${sec}`;
    }

    updateStatus(text, classes, dotClass) {
        this.statusBadge.innerHTML = `<div class="w-2.5 h-2.5 rounded-full ${dotClass} status-pulse"></div> ${text}`;
        this.statusBadge.className = `px-6 py-3 rounded-full border font-black uppercase tracking-widest text-xs flex items-center gap-3 ${classes}`;
    }

    downloadSessionData() {
        const sessionData = {
            session_id: Date.now(),
            duration: this.timerEl.innerText,
            violations: this.violations,
            average_focus: this.attentionScoreEl.innerText,
            pomo_sessions: this.pomoMode,
            timestamp: new Date().toISOString()
        };
        const blob = new Blob([JSON.stringify(sessionData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `TELEMETRY_${Date.now()}.json`;
        a.click();
    }
}

document.addEventListener('DOMContentLoaded', () => { window.sentinel = new FocusSentinel(); });

