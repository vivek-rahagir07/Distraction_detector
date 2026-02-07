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
        this.audioVisualizer = document.getElementById('audio_visualizer');
        this.audioCanvasCtx = this.audioVisualizer ? this.audioVisualizer.getContext('2d') : null;

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
        this.analyser = null;
        this.lastObjectResults = null;

        // INVIGILATION_STATE
        this.riskIndex = 0;
        this.isInducting = false;
        this.tabViolations = 0;

        // MEDIAPIPE_INIT
        this.faceMesh = new FaceMesh({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
        });

        this.faceMesh.setOptions({
            maxNumFaces: 10,
            refineLandmarks: true,
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.6
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
        this.lastObjectResults = results;
        if (!results.detections) return;

        const forbidden = ['cell phone', 'laptop', 'tablet', 'book', 'person']; // person is for multi-person logic if needed
        for (const detection of results.detections) {
            const category = detection.categories[0].categoryName;
            if (forbidden.includes(category) && category !== 'person') {
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

        // TAB_MONITORING
        window.addEventListener('blur', () => this.handleTabViolation('BROWSER_UNFOCUSED'));
        window.addEventListener('focus', () => this.addLog('BROWSER_RE-FOCUSED', 'success'));
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) this.handleTabViolation('WINDOW_HIDDEN');
        });

        this.initChart();
        this.runDiagnostics();
        console.log("SYS_INIT_OK_V3.0");
    }

    async runDiagnostics() {
        const diagStatus = document.getElementById('diagStatus');
        const checkItems = {
            camera: document.getElementById('check-camera'),
            mic: document.getElementById('check-mic'),
            models: document.getElementById('check-models'),
            network: document.getElementById('check-network')
        };

        const updateItem = (id, success) => {
            const el = checkItems[id];
            if (!el) return;
            el.className = `flex items-center gap-2 ${success ? 'diag-item-ok' : 'diag-item-err'}`;
        };

        diagStatus.innerText = "INITIALIZING...";

        // 1. Network Check
        updateItem('network', navigator.onLine);
        window.addEventListener('online', () => updateItem('network', true));
        window.addEventListener('offline', () => updateItem('network', false));

        // 2. Camera Check
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            stream.getTracks().forEach(t => t.stop());
            updateItem('camera', true);
        } catch { updateItem('camera', false); }

        // 3. Mic Check
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            stream.getTracks().forEach(t => t.stop());
            updateItem('mic', true);
        } catch { updateItem('mic', false); }

        // 4. Neural Engines (Wait for ObjectDetector)
        let modelRetries = 0;
        const modelCheck = setInterval(() => {
            if (this.objectDetector) {
                updateItem('models', true);
                diagStatus.innerText = "SYSTEM_READY";
                diagStatus.className = "text-emerald-500 animate-none";
                this.requestBtn.disabled = false;
                this.requestBtn.classList.remove('opacity-50');
                clearInterval(modelCheck);
            } else if (modelRetries > 10) {
                updateItem('models', false);
                diagStatus.innerText = "MODEL_ERROR";
                clearInterval(modelCheck);
            }
            modelRetries++;
        }, 1000);
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
        if (this.isInducting) return;
        const btnMainText = document.getElementById('btnMainText');
        btnMainText.innerText = "AUTHENTICATING...";
        this.requestBtn.disabled = true;

        try {
            await this.camera.start();
            await this.startAudioAnalysis();

            // Start Induction Phase
            this.startInduction();
        } catch (err) {
            console.error(err);
            const errorMsg = document.getElementById('errorMessage');
            errorMsg.innerText = "SECURITY_ERR: ACCESS_DENIED";
            errorMsg.classList.remove('hidden');
            btnMainText.innerText = "RE-INITIALIZE";
            this.requestBtn.disabled = false;
        }
    }

    async startInduction() {
        this.isInducting = true;
        document.querySelector('.text-left.bg-black\\/40').classList.add('hidden');
        document.getElementById('inductionPhase').classList.remove('hidden');

        let progress = 0;
        const progressBar = document.getElementById('inductionProgress');

        this.playSuccess();
        this.addLog("BIOMETRIC_INDUCTION_STARTED", "info");

        const inductionInterval = setInterval(() => {
            progress += 2;
            progressBar.style.width = `${progress}%`;

            if (progress >= 100) {
                clearInterval(inductionInterval);
                this.finishSystemActivation();
            }
        }, 100);
    }

    finishSystemActivation() {
        this.isInducting = false;
        this.isActive = true;
        this.setupOverlay.classList.add('hidden');
        this.stopBtn.classList.remove('hidden');
        this.updateStatus("SECURED", "border-red-500 text-red-500 bg-red-500/10", "bg-red-500");
        this.startTimer();

        this.canvasElement.width = this.videoElement.videoWidth || 640;
        this.canvasElement.height = this.videoElement.videoHeight || 480;

        if (this.audioVisualizer) {
            this.audioVisualizer.width = 600;
            this.audioVisualizer.height = 100;
        }

        this.addLog("IDENTITY_LOCKED_SECURE", "success");
        this.playSuccess();
        this.startPomo();
    }

    async startAudioAnalysis() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.initAudio();
            const source = this.audioCtx.createMediaStreamSource(stream);
            this.analyser = this.audioCtx.createAnalyser();
            this.analyser.fftSize = 512;
            source.connect(this.analyser);

            const dataArray = new Uint8Array(this.analyser.frequencyBinCount);
            this.audioInterval = setInterval(() => {
                this.analyser.getByteFrequencyData(dataArray);
                const average = dataArray.reduce((p, c) => p + c, 0) / dataArray.length;
                if (average > 45) { // Threshold for talking/noise
                    this.handleViolation("VOCAL_DETECTION");
                }
            }, 500);
        } catch (err) {
            console.warn("Audio monitoring disabled: ", err);
        }
    }

    drawAudioWave() {
        if (!this.analyser || !this.audioCanvasCtx) return;
        const ctx = this.audioCanvasCtx;
        const bufferLength = this.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        this.analyser.getByteTimeDomainData(dataArray);

        ctx.clearRect(0, 0, this.audioVisualizer.width, this.audioVisualizer.height);
        ctx.lineWidth = 2;
        ctx.strokeStyle = '#00f3ff';
        ctx.beginPath();

        const sliceWidth = this.audioVisualizer.width * 1.0 / bufferLength;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            const v = dataArray[i] / 128.0;
            const y = v * this.audioVisualizer.height / 2;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
            x += sliceWidth;
        }

        ctx.lineTo(this.audioVisualizer.width, this.audioVisualizer.height / 2);
        ctx.stroke();
    }

    onResults(results) {
        this.canvasCtx.save();
        this.canvasCtx.clearRect(0, 0, this.canvasElement.width, this.canvasElement.height);
        this.canvasCtx.drawImage(results.image, 0, 0, this.canvasElement.width, this.canvasElement.height);

        this.framesActive++;
        this.drawAudioWave();

        // 1. Draw Object Detections
        if (this.lastObjectResults && this.lastObjectResults.detections) {
            this.lastObjectResults.detections.forEach(detection => {
                const category = detection.categories[0].categoryName;
                if (['cell phone', 'laptop', 'tablet', 'book'].includes(category)) {
                    this.drawObjectBracket(detection);
                }
            });
        }

        // 2. Face Analytics
        let distractedCount = 0;
        const faceCount = results.multiFaceLandmarks ? results.multiFaceLandmarks.length : 0;

        if (faceCount > 0) {
            results.multiFaceLandmarks.forEach((landmarks, index) => {
                const distractionType = this.checkLookingAway(landmarks);
                const isDistracted = distractionType !== null;
                if (isDistracted) distractedCount++;

                const color = isDistracted ? '#ef4444' : '#22c55e';
                drawConnectors(this.canvasCtx, landmarks, FACEMESH_TESSELATION, { color: `${color}10`, lineWidth: 0.5 });
                this.drawPersonBrackets(landmarks, color, isDistracted);
                this.drawPersonLabels(landmarks, index, distractionType, color);
            });

            if (distractedCount > 0) {
                this.handleViolation("GAZE_DEV");
                this.updateUIForDistraction(true, distractedCount > 1 ? "MULTIPLE_GAZE_DEV" : "SINGLE_GAZE_DEV");
            } else {
                this.framesFocused++;
                this.updateUIForDistraction(false);
            }
        } else {
            this.handleViolation("SIG_LOST");
            this.updateUIForDistraction(true, "SIGNAL_LOST");
        }

        this.updateMetrics();
        this.canvasCtx.restore();
    }

    drawObjectBracket(detection) {
        const { originX, originY, width, height } = detection.boundingBox;
        const ctx = this.canvasCtx;

        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(originX, originY, width, height);
        ctx.setLineDash([]);

        ctx.fillStyle = '#ef4444';
        ctx.font = 'bold 10px Orbitron';
        ctx.fillText(`TARGET: ${detection.categories[0].categoryName.toUpperCase()}`, originX, originY - 5);
    }

    drawPersonLabels(landmarks, index, distractionType, color) {
        const topLandmark = landmarks[10];
        const x = topLandmark.x * this.canvasElement.width;
        const y = topLandmark.y * this.canvasElement.height - 60;

        this.canvasCtx.fillStyle = color;
        this.canvasCtx.font = 'bold 10px Orbitron';
        this.canvasCtx.textAlign = 'center';
        this.canvasCtx.fillText(`AGENT_00${index + 1}`, x, y - 10);

        this.canvasCtx.font = 'bold 14px Rajdhani';
        if (distractionType) {
            const boxW = 160;
            const boxH = 24;
            this.canvasCtx.fillStyle = '#ef4444';
            this.canvasCtx.fillRect(x - boxW / 2, y - boxH / 2, boxW, boxH);
            this.canvasCtx.fillStyle = '#ffffff';

            let label = '⚠ FOCUS BROKEN';
            if (distractionType === 'DESK_GAZE') label = '⚠ DESK_GAZE_DETECTED';
            if (distractionType === 'SIDE_GAZE') label = '⚠ SIDE_GAZE_DETECTED';

            this.canvasCtx.fillText(label, x, y + 5);
        } else {
            this.canvasCtx.fillText('✓ SECURED', x, y + 5);
        }
    }

    drawPersonBrackets(landmarks, color, isDistracted) {
        // Calculate bounding box from landmarks
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        landmarks.forEach(l => {
            minX = Math.min(minX, l.x);
            minY = Math.min(minY, l.y);
            maxX = Math.max(maxX, l.x);
            maxY = Math.max(maxY, l.y);
        });

        const padding = 0.05;
        const w = this.canvasElement.width;
        const h = this.canvasElement.height;

        const left = (minX - padding) * w;
        const top = (minY - padding) * h;
        const right = (maxX + padding) * w;
        const bottom = (maxY + padding) * h;
        const width = right - left;
        const height = bottom - top;
        const cornerSize = width * 0.2;

        this.canvasCtx.strokeStyle = color;
        this.canvasCtx.lineWidth = isDistracted ? 3 : 2;
        this.canvasCtx.beginPath();

        // Top-Left
        this.canvasCtx.moveTo(left, top + cornerSize);
        this.canvasCtx.lineTo(left, top);
        this.canvasCtx.lineTo(left + cornerSize, top);

        // Top-Right
        this.canvasCtx.moveTo(right - cornerSize, top);
        this.canvasCtx.lineTo(right, top);
        this.canvasCtx.lineTo(right, top + cornerSize);

        // Bottom-Right
        this.canvasCtx.moveTo(right, bottom - cornerSize);
        this.canvasCtx.lineTo(right, bottom);
        this.canvasCtx.lineTo(right - cornerSize, bottom);

        // Bottom-Left
        this.canvasCtx.moveTo(left + cornerSize, bottom);
        this.canvasCtx.lineTo(left, bottom);
        this.canvasCtx.lineTo(left, bottom - cornerSize);

        this.canvasCtx.stroke();

        // Subtle glow effect
        if (isDistracted) {
            this.canvasCtx.shadowBlur = 15;
            this.canvasCtx.shadowColor = color;
            this.canvasCtx.stroke();
            this.canvasCtx.shadowBlur = 0;
        }
    }

    checkLookingAway(landmarks) {
        const leftEye = landmarks[33];
        const rightEye = landmarks[263];
        const nose = landmarks[1];

        // Horizontal Ratio (X)
        const eyeDist = rightEye.x - leftEye.x;
        const noseRelativeX = (nose.x - leftEye.x) / eyeDist;

        // Vertical Ratio (Y)
        const forehead = landmarks[10];
        const chin = landmarks[152];
        const faceHeight = chin.y - forehead.y;
        const noseRelativeY = (nose.y - forehead.y) / faceHeight;

        // Depth Check (Z) - detects extreme head turn
        const zRatio = Math.abs(leftEye.z - rightEye.z) / eyeDist;

        const isXAway = noseRelativeX < 0.35 || noseRelativeX > 0.65;
        const isYAway = noseRelativeY < 0.35 || noseRelativeY > 0.65;
        const isZAway = zRatio > 0.35;

        if (isXAway || isZAway) return 'SIDE_GAZE';
        if (isYAway) {
            return (noseRelativeY > 0.65) ? 'DESK_GAZE' : 'HIGH_GAZE';
        }

        return null;
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
            this.updateRisk(15); // Increase risk by 15% per violation
        }
    }

    handleTabViolation(type) {
        if (!this.isActive) return;
        this.tabViolations++;
        this.handleViolation(`${type}_ID_${this.tabViolations}`);
        this.addLog(`SECURITY_BREACH: ${type} - TAB_RECORDED`, "alert");
    }

    updateRisk(amount) {
        this.riskIndex = Math.min(100, this.riskIndex + amount);
        const meter = document.getElementById('riskMeter');
        const label = document.getElementById('riskLabel');

        meter.style.width = `${this.riskIndex}%`;

        if (this.riskIndex > 70) {
            label.innerText = 'CRITICAL_RISK';
            label.className = 'text-[9px] font-mono text-red-500 animate-pulse';
        } else if (this.riskIndex > 30) {
            label.innerText = 'MODERATE_RISK';
            label.className = 'text-[9px] font-mono text-amber-500';
        } else {
            label.innerText = 'LOW_RISK';
            label.className = 'text-[9px] font-mono text-emerald-500';
        }
    }

    updateUIForDistraction(isDistracted, customState = null) {
        if (isDistracted) {
            this.updateStatus(customState || "SECURITY_ALERT", "border-red-500 text-red-500 bg-red-500/10", "bg-red-500");
            this.alertOverlay.style.opacity = "1";
            this.alertOverlay.style.borderColor = "rgba(239, 68, 68, 0.4)";
            this.videoFrame.classList.add('neon-border-red');
            this.videoFrame.classList.remove('neon-border-green', 'neon-border-cyan');
            this.focusStateEl.innerText = customState || "INTEGRITY_COMPROMISED";
            this.focusStateEl.style.color = '#ef4444';
            this.focusStateEl.className = "text-xl font-bold uppercase glow-text-red animate-pulse";
        } else {
            this.updateStatus("SECURED", "border-emerald-500 text-emerald-400 bg-emerald-500/10", "bg-emerald-500");
            this.alertOverlay.style.opacity = "0";
            this.alertOverlay.style.borderColor = "rgba(239, 68, 68, 0)";
            this.videoFrame.classList.add('neon-border-green');
            this.videoFrame.classList.remove('neon-border-red', 'neon-border-cyan');
            this.focusStateEl.innerText = "COMPLIANT";
            this.focusStateEl.style.color = '#10b981';
            this.focusStateEl.className = "text-xl font-bold uppercase";
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

        // Dynamic color for Attention Score
        if (ratio > 85) this.attentionScoreEl.style.color = '#10b981';
        else if (ratio > 60) this.attentionScoreEl.style.color = '#34d399';
        else if (ratio > 40) this.attentionScoreEl.style.color = '#fbbf24';
        else this.attentionScoreEl.style.color = '#ef4444';

        if (this.framesActive % 20 === 0) {
            this.chart.data.datasets[0].data.shift();
            this.chart.data.datasets[0].data.push(ratio);
            this.chart.update('none');

            const consistency = Math.round(ratio);
            const scoreLabel = document.getElementById('consistencyScore');
            const scoreBar = document.getElementById('consistencyBar');

            scoreLabel.innerText = consistency + "%";
            scoreBar.style.width = consistency + "%";

            // Industrial color mapping for bar
            if (consistency > 80) scoreBar.className = "bg-emerald-500 h-full transition-all duration-500";
            else if (consistency > 50) scoreBar.className = "bg-amber-500 h-full transition-all duration-500";
            else scoreBar.className = "bg-red-500 h-full transition-all duration-500";
        }
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

