# YG Aimbot

<div align="center">
  <img src="data/aim.svg" alt="YG Aimbot Logo" width="100"/>

  <p><b>AI-powered aim assistant for FPS games</b><br/>
  <i>实时推理 · 精准瞄准 · 可视化配置</i></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10"/>
    <img src="https://img.shields.io/badge/CUDA-12.8-76B900?logo=nvidia&logoColor=white" alt="CUDA 12.8"/>
    <img src="https://img.shields.io/badge/TensorRT-10.9-FF6F00?logo=nvidia&logoColor=white" alt="TensorRT 10.9"/>
    <img src="https://img.shields.io/badge/ONNX-1.23-005CED?logo=onnx&logoColor=white" alt="ONNX 1.23"/>
    <img src="https://img.shields.io/badge/Electron-47848F?logo=electron&logoColor=white" alt="Electron"/>
  </p>
</div>

---

## Architecture

Five threads coordinate the capture → inference → track → aim pipeline:

```
                        Main Loop (200Hz)

    Capture ──▶ Inference ──▶ Tracker ──▶ Aim ────▶ Mouse
    
                                              │
                                         UIBridge
                                              │
                                         WebSocket :8765
                                              │
                                         Electron GUI
```

| Thread | Role |
|--------|------|
| **Main loop** | Coordinates pipeline at 200Hz hotkey check rate |
| **Capture** | Screen capture via `mss` |
| **Inference** | Single worker — YOLO (TensorRT) or ONNX Runtime |
| **Mouse move** | Adaptive-fraction movement at ~2000Hz |
| **WebSocket** | Event loop for UI communication |

Inference supports two backends: `.pt`/`.engine` via `ultralytics.YOLO` with TensorRT optimisation, and `.onnx` via ONNX Runtime with custom pre/post-processing. The tracker selects the nearest valid target using vectorised numpy, and extrapolates through detection gaps (up to 150ms) with a velocity-weighted linear predictor. Mouse deltas go through body offset → smoothing → tremor → DPI mapping before being applied by a dedicated worker thread.

Three hotkey modes: **hold** (while pressed), **toggle** (press on/off), **auto** (always on). Model hot-reload resizes the capture window automatically.

---

## Quick Start

### Prerequisites

| | |
|----------|---------|
| **Python** | 3.10 |
| **CUDA**  | 12.8 |
| **TensorRT** | 10.9 |
| **Node.js** | 18+ |

### Install

```bash
# Python environment
conda create -n yg_aimbot python=3.10
conda activate yg_aimbot
pip install --force-reinstall -r requirements.txt
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install ./tensorrt-10.9.0.34-cp310-none-win_amd64.whl
pip install numpy==1.26.4

# Electron UI
cd electron_ui && npm install && cd ..
```

### Run

```bash
python run.py
```

### First-Time Setup

1. Place model files (`.pt` / `.engine` / `.onnx`) in `data/`
2. Launch → select model in **AI Config** → configure hotkeys in **Aim Config**
3. Adjust `body_x_offset` / `body_y_offset` using the drag canvas
4. Launch your game and test

---

## Configuration

Configuration lives in `config.ini`. All settings are editable through the GUI, and most take effect immediately — no restart needed.

### `[AI]` — Model & Detection

| Key | Default | Description |
|-----|---------|-------------|
| `model_name` | `YOLOv5s_apex_320.onnx` | Model file in `data/` |
| `conf` | `0.2` | Confidence threshold |
| `device` | `0` | CUDA device ID |

### `[Capture]` — Screen Capture

| Key | Default | Description |
|-----|---------|-------------|
| `window_width` | `320` | Capture width (auto-set to model input) |
| `window_height` | `320` | Capture height |
| `fps` | `240` | Capture frame rate limit |
| `circle` | `False` | Circular mask on capture region |
| `ai_debug` | `False` | Detection overlay + video stream in UI |

### `[Aim]` — Aim Behaviour

| Key | Default | Description |
|-----|---------|-------------|
| `auto` | `False` | Always-on mode |
| `mode` | `hold` | `hold` / `toggle` |
| `target_cls` | `0` | Class to target (0 = enemy, 1 = teammate) |
| `body_x_offset` | `-0.03` | Horizontal aim offset (fraction of width) |
| `body_y_offset` | `-0.41` | Vertical aim offset (fraction of height) |
| `hotkeys` | `X1MouseButton,...` | Comma-separated hotkey names |
| `max_target_distance` | `100` | Max engagement distance (px) |
| `max_miss_time` | `0.15` | Max blind prediction time (s) |
| `max_miss_distance` | `120` | Max blind prediction distance (px) |

### `[Mouse]` — Mouse Output

| Key | Default | Description |
|-----|---------|-------------|
| `dpi` | `2400` | Mouse DPI |
| `sensitivity` | `5` | In-game sensitivity |
| `fov_width` | `40` | Horizontal FOV (degrees) |
| `fov_height` | `40` | Vertical FOV (degrees) |

---

## Project Structure

```
yg_aimbot/
├── run.py                          # Entry point
├── config.ini                      # Persistent config
├── data/                           # Model files (.pt / .engine / .onnx)
│
├── core/
│   ├── aimbot.py                   # Main loop, hotkey state machine
│   ├── buttons.py                  # Win32 virtual key codes
│   ├── logger.py                   # Coloured logging + WebSocket handler
│   ├── ui_bridge.py                # UI event dispatch
│   ├── websocket_server.py         # WebSocket server (port 8765)
│   │
│   └── services/
│       ├── config_service.py       # INI config with callback registry
│       ├── capture_service.py      # Screen capture (mss)
│       ├── inference_service.py    # Async YOLO / ONNX inference
│       ├── tracker_service.py      # Target selection + velocity prediction
│       └── aim_service.py          # Mouse delta + adaptive movement
│
└── electron_ui/
    ├── main.js                     # Electron window setup
    ├── index.html                  # Vue 2 + Element UI SPA
    ├── app.js                      # WebSocket client + config forms
    └── package.json                # Electron packaging
```

---

## FAQ

<details>
<summary><b>Is this detectable by anti-cheat systems?</b></summary>
We can't make guarantees. Mouse movement uses a hardware-level driver (<code>makcu</code>), but no technique is undetectable. Use at your own risk.
</details>

<details>
<summary><b>Can I use this with other games?</b></summary>
The included models are trained for Apex Legends. You'd need a custom YOLO model for other games.
</details>

<details>
<summary><b>How do I improve accuracy?</b></summary>
Start with <code>body_x_offset</code> / <code>body_y_offset</code> using the drag canvas, then tune smoothing and prediction to match target movement.
</details>

<details>
<summary><b>What's the difference between model formats?</b></summary>
<code>.pt</code> = raw PyTorch (slowest). <code>.engine</code> = TensorRT-optimised (fastest). <code>.onnx</code> = ONNX Runtime. Format is auto-detected from the file extension.
</details>

---

## Disclaimer

This software is provided for **educational and research purposes only**. Using aim-assistance in online multiplayer games may violate the game's Terms of Service and result in account bans. You assume all responsibility.
