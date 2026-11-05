/* ══════════════════════════════════════════════════
   ISL Bridge – script.js (enhanced)
   ══════════════════════════════════════════════════ */

"use strict";

// ── State ─────────────────────────────────────────
let isRunning = false;
let currentStream = null;
let videoEl = null;
let currentAudio = null;
let frameIntervalId = null;  // used for rAF-loop control

// ── Boot ──────────────────────────────────────────
window.addEventListener("load", () => {
    // Load language list
    fetch("/languages")
        .then(r => r.json())
        .then(data => {
            const sel = document.getElementById("language");
            Object.entries(data)
                .filter(([, code]) => code !== "en")
                .forEach(([name, code]) => {
                    const opt = document.createElement("option");
                    opt.value = code;
                    opt.textContent = name;
                    sel.appendChild(opt);
                });
        });

    // Reset server state on load
    fetch("/reset");
});

// ── Setup-screen: input source selection ──────────
document.getElementById("camera-button").addEventListener("change", () => {
    document.getElementById("file-input").style.display = "none";
    document.getElementById("start").style.display = "block";
    document.getElementById("prompt").style.display = "none";
});

document.getElementById("video-button").addEventListener("change", () => {
    document.getElementById("file-input").style.display = "block";
    document.getElementById("start").style.display = "none";
});

document.getElementById("video-file").addEventListener("change", () => {
    if (document.getElementById("video-file").files[0]) {
        document.getElementById("start").style.display = "block";
    }
});

// ── Start button ──────────────────────────────────
document.getElementById("start").addEventListener("click", () => {
    const videoSelected = document.getElementById("video-button").checked;

    if (videoSelected && !document.getElementById("video-file").files[0]) {
        const p = document.getElementById("prompt");
        p.style.display = "block";
        p.textContent = "Please select a video file";
        return;
    }

    // Transition to main screen
    document.getElementById("setup-screen").style.display = "none";
    const ms = document.getElementById("main-screen");
    ms.style.display = "flex";

    if (!videoSelected) {
        // ── Camera mode
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                videoEl = document.getElementById("video-feed");
                videoEl.srcObject = stream;
                currentStream = stream;
                videoEl.addEventListener("loadeddata", () => {
                    isRunning = true;
                    sendFrames(videoEl);
                    document.getElementById("video-hint").style.display = "none";
                }, { once: true });
            })
            .catch(err => console.error("Camera error:", err));
    } else {
        // ── Video file mode
        document.getElementById("pause").style.display = "inline-block";
        const file = document.getElementById("video-file").files[0];
        const url = URL.createObjectURL(file);
        videoEl = document.getElementById("video-feed");
        videoEl.src = url;
        videoEl.addEventListener("loadeddata", () => {
            isRunning = true;
            sendFrames(videoEl);
            document.getElementById("video-hint").style.display = "none";
        }, { once: true });
        videoEl.addEventListener("ended", () => { isRunning = false; });
    }
});

// ── Language change ───────────────────────────────
document.getElementById("language").addEventListener("change", function () {
    fetch("/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: this.value }),
    })
        .then(r => r.json())
        .then(data => {
            document.getElementById("sentence").textContent = data || "";
        });
});

// ── Reset ALL ─────────────────────────────────────
document.getElementById("reset").addEventListener("click", () => {
    fetch("/reset").then(() => {
        clearDisplay();
        clearHistory();
        if (!currentStream) {
            // Video file mode
            isRunning = false;
            videoEl.pause();
            videoEl.currentTime = 0;
            const pb = document.getElementById("pause");
            pb.textContent = "▶ Resume";
            showPrompt2("Click Resume to start the video");
        }
    });
});

// ── Reset current word only ───────────────────────
document.getElementById("reset-word").addEventListener("click", () => {
    fetch("/reset_word").then(() => {
        document.getElementById("word-formed-so-far").textContent = "–";
    });
});

// ── End session ───────────────────────────────────
document.getElementById("end").addEventListener("click", () => {
    isRunning = false;
    if (currentStream) currentStream.getTracks().forEach(t => t.stop());
    if (videoEl) { videoEl.pause(); videoEl.src = ""; }
    fetch("/reset");
    document.getElementById("main-screen").style.display = "none";
    const gs = document.getElementById("goodbye-screen");
    gs.style.display = "flex";
});

// ── Speak ─────────────────────────────────────────
document.getElementById("speak").addEventListener("click", () => {
    const code = document.getElementById("language").value;
    fetch("/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code }),
    })
        .then(r => (r.ok ? r.blob() : null))
        .then(blob => {
            if (!blob) return;
            if (currentAudio) currentAudio.pause();
            const url = URL.createObjectURL(blob);
            currentAudio = new Audio(url);
            currentAudio.play();
        });
});

// ── Copy sentence ─────────────────────────────────
document.getElementById("copy-btn").addEventListener("click", () => {
    const text = document.getElementById("sentence").textContent;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById("copy-btn");
        const orig = btn.textContent;
        btn.textContent = "✓ Copied";
        setTimeout(() => { btn.textContent = orig; }, 1500);
    });
});

// ── Pause / Resume (video mode) ───────────────────
document.getElementById("pause").addEventListener("click", function () {
    if (isRunning) {
        isRunning = false;
        videoEl.pause();
        this.textContent = "▶ Resume";
    } else {
        document.getElementById("prompt2").style.display = "none";
        isRunning = true;
        videoEl.play();
        sendFrames(videoEl);
        this.textContent = "⏸ Pause";
    }
});

// ══════════════════════════════════════════════════
//  FRAME LOOP
// ══════════════════════════════════════════════════
function sendFrames(video) {
    if (!isRunning) return;

    const canvas = document.getElementById("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
        setTimeout(() => { if (isRunning) sendFrames(video); }, 500);
        return;
    }

    canvas.getContext("2d").drawImage(video, 0, 0);
    const imageData = canvas.toDataURL("image/jpeg", 0.8);  // 80% quality – faster

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            image: imageData,
            language: document.getElementById("language").value,
        }),
    })
        .then(r => r.json())
        .then(data => {
            updateDisplay(data);
            if (isRunning) sendFrames(video);
        })
        .catch(() => {
            if (isRunning) sendFrames(video);
        });
}

// ══════════════════════════════════════════════════
//  UI HELPERS
// ══════════════════════════════════════════════════

function updateDisplay(data) {
    // Sign glyph
    const signEl = document.getElementById("confirmed-sign");
    signEl.textContent = data.sign || "–";

    // Confidence meter
    const conf = data.confidence || 0;
    document.getElementById("conf-bar").style.width = `${Math.round(conf * 100)}%`;
    document.getElementById("conf-pct").textContent = `${Math.round(conf * 100)}%`;

    // Hold-progress meter
    const hold = data.hold_progress || 0;
    document.getElementById("hold-bar").style.width = `${Math.round(hold * 100)}%`;
    document.getElementById("hold-pct").textContent = `${Math.round(hold * 100)}%`;

    // Flash the sign glyph when a sign is fully confirmed (hold == 1)
    if (hold >= 1 && data.sign) {
        signEl.classList.add("sign-flash");
        setTimeout(() => signEl.classList.remove("sign-flash"), 300);
    }

    // Word
    document.getElementById("word-formed-so-far").textContent = data.word || "–";

    // Sentence
    document.getElementById("sentence").textContent = data.sentence || "";

    // Refresh sign history from server occasionally
    // (we do it every time a new confirmed sign arrives)
    if (data.sign) refreshHistory();
}

function clearDisplay() {
    document.getElementById("confirmed-sign").textContent = "–";
    document.getElementById("word-formed-so-far").textContent = "–";
    document.getElementById("sentence").textContent = "";
    document.getElementById("conf-bar").style.width = "0%";
    document.getElementById("conf-pct").textContent = "0%";
    document.getElementById("hold-bar").style.width = "0%";
    document.getElementById("hold-pct").textContent = "0%";
}

function clearHistory() {
    const hs = document.getElementById("history-scroll");
    hs.innerHTML = '<span class="history-empty">No signs confirmed yet</span>';
}

let _lastHistoryLen = 0;
function refreshHistory() {
    fetch("/history")
        .then(r => r.json())
        .then(items => {
            if (items.length === _lastHistoryLen) return;
            _lastHistoryLen = items.length;

            const hs = document.getElementById("history-scroll");
            hs.innerHTML = "";

            if (!items.length) {
                hs.innerHTML = '<span class="history-empty">No signs confirmed yet</span>';
                return;
            }

            items.forEach(item => {
                const chip = document.createElement("div");
                chip.className = "history-chip";
                chip.title = `Confidence: ${Math.round(item.confidence * 100)}%`;
                chip.innerHTML = `
                    <span class="chip-letter">${item.sign}</span>
                    <span class="chip-conf">${Math.round(item.confidence * 100)}%</span>
                `;
                hs.appendChild(chip);
            });

            // Auto-scroll to newest
            hs.scrollLeft = hs.scrollWidth;
        });
}

function showPrompt2(msg) {
    const p = document.getElementById("prompt2");
    p.textContent = msg;
    p.style.display = "block";
}
