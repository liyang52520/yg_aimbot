# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

- **Python 3.10** conda environment named `yg_aimbot` — activate with `conda activate yg_aimbot` before running anything
- Python dependencies: `pip install -r requirements.txt`
- Electron UI dependencies: `cd electron_ui && npm install`
- Models go in the `data/` directory
- CUDA 12.8 + TensorRT 10.9 required for GPU inference

## Commands

- **Run application**: `python run.py` (launches Python core + WebSocket + Electron UI)
- **Run Electron UI standalone**: `cd electron_ui && npm start` (requires Python core running separately)
- **Install Python deps**: `pip install --force-reinstall -r requirements.txt`
- **No test suite yet** — only `.pyc` artifacts exist in `core/testing/`

## Architecture

### Threading Model

| Thread | Purpose | Sync Mechanism |
|--------|---------|---------------|
| **Main async loop** (`Aimbot._main_loop`) | Coordinates: capture frame → submit inference → wait result → tracker update | `asyncio`, `run_in_executor` for blocking wait |
| **Capture thread** (`ScreenCaptureService`) | Screen capture via `mss` at configurable FPS | Thread-safe lock for frame access |
| **Inference worker** (`AsyncInferenceService._inference_worker`) | Runs YOLO/ONNX inference | `threading.Event` (frame_event / result_event) |
| **Mouse move worker** (`AimService._move_worker`) | Adaptive-fraction mouse movement via `makcu` | Lock-protected target state, bounded loop |
| **WebSocket** (`WebSocketServer`) | asyncio event loop in its own thread | Thread-safe message queue + `run_coroutine_threadsafe` |

### Data Flow Pipeline

```
CaptureService ──frame──► InferenceService ──result──► TrackerService ──target──► AimService ──delta──► MouseControlService
     ▲                       │                                                    │
     └───── config callbacks ┘                                                    │
                                                                                  │
     UIBridge ◄───────────────────────────────────────────────────────────────────┘
        │
        ▼
  WebSocket (port 8765) ◄────► Electron UI (Vue 2 + Element UI, CDN, no build)
```

All services are **singletons** instantiated at module level. Config changes propagate to all services via a callback registration system (`ConfigService.register_callback`). Each service registers a callback that caches its relevant config values locally to avoid lock contention on hot paths.

### Main Loop (`core/aimbot.py`)

The async `_main_loop` runs at 200Hz hotkey check rate:
1. Check hotkey state → determines `_need_prediction`
2. If predicting: submit latest frame → wait up to 5ms for inference result → update tracker → send detections/FPS to UI via `UIBridge`
3. If idle: sleep 5ms to avoid CPU spin
4. Periodic config check every 200ms (model change, capture config, aim config)
5. On inference-fps drops to 0 when prediction stops → notifies UI to clear display

### Hotkey State Machine

Three modes controlled by `[aim] mode` and `[aim] auto`:
- **hold**: Check `win32api.GetKeyState` for any hotkey being pressed. Active while any hotkey is held.
- **toggle**: Edge-detect key press/release → toggle `_toggle_enabled` state. Active until toggled off.
- **auto** (`auto = true`): Always active, ignores hotkey state.

Hotkeys are parsed from comma-separated string in config, mapped to Win32 virtual key codes via `Buttons.KEY_CODES` (`core/buttons.py`).

### Model System

Two backends selected by file extension:
- **`.pt` / `.engine`** → `ultralytics.YOLO()` with TensorRT optimization
- **`.onnx`** → `onnxruntime.InferenceSession` with custom pre/post-processing

Model hot-reload: When `_check_config` detects `model_name` change, `inference_service.reload()` calls stop → unload → load → start, and capture window is resized to match the new model's input size.

### Config System (`core/services/config_service.py`)

- INI-based (`config.ini`), `Lock`-protected reads/writes
- `DEFAULT_CONFIG` dict provides fallback defaults for all keys
- Sections: `[AI]`, `[Capture]`, `[Aim]`, `[Mouse]`
- Callbacks: `register_callback(section, updates)` → all services update their cached config values
- `update_section(notify=True)` triggers callbacks for live config changes
- Saving writes the cached values back to `config.ini`

### Target Tracking (`core/services/tracker_service.py`)

- `TargetPredictor` maintains velocity-based linear extrapolation from last 5 detections
- During detection gaps (up to `max_miss_time` = 150ms), predicts target position using weighted-average velocity
- Falls back to `_handle_no_detection` which checks max_miss_distance and resets prediction if exceeded
- Target selection: pure numpy, chooses nearest detection to screen center filtered by `target_cls`
- Adaptive tracking distance scales with tracked target size

### Aim Service (`core/services/aim_service.py`)

Mouse delta computation pipeline:
```
body offset → model-space centering → velocity prediction → smoothing → tremor → DPI/sensitivity → clamp
```

- **Adaptive fraction movement**: exponential curve mapping remaining distance to move fraction (0.25 at zero, 0.55 at infinity) — far targets converge fast, near targets are precise
- **Sub-pixel accumulation**: `_frac_x` / `_frac_y` accumulators prevent precision loss from `int()` truncation
- **Frame move cap**: `max_frame_move` (8.0 count/frame) prevents teleportation
- Uses `makcu` hardware mouse controller (`create_controller(auto_reconnect=True)`)

### WebSocket Protocol (`core/websocket_server.py`)

**Commands** (client → server):
| Command | Payload | Description |
|---------|---------|-------------|
| `get-config` | — | Sends full config to client |
| `save-config` | `{data: {section: {key: val}}}` | Saves to config.ini + persists to disk |
| `apply-config` | `{data: {section: {key: val}}}` | Applies to memory only (no disk write) |
| `scan-models` | — | Scans `data/` dir for .pt/.onnx/.engine files |

**Push types** (server → client):
| Type | Data | Frequency |
|------|------|-----------|
| `config` | Full `{ai, capture, aim, mouse}` section dicts | On connect + model reload |
| `fps` | `{capture: float, predict: float}` | ~10Hz |
| `video-frame` | JPEG base64 string, max 640px wide | Max 30fps |
| `detections` | `[{box, confidence, class_id}]` | Per inference result |
| `log` | String | Per log line |
| `config-saved` | `{success: bool}` | After save-config |
| `models` | `[filename, ...]` | After scan-models |

### UI (`electron_ui/`)

- Vue.js 2 SPA (CDN-loaded, no build step) in `app.js`
- Element UI component library (CDN)
- Config tabs map 1:1 to `config.ini` sections
- Hotkey multiselect, body offset drag canvas, video stream with detection overlay
- Electron entry point: `main.js`, packaged via `electron-builder`

### Frame ID Scheme

Snowflake-like: `(current_second << 20) | counter` — monotonically increasing, enables deduplication across services.

### Logging (`core/logger.py`)

- `ColoredFormatter`: ANSI-colored console output per log level
- `WebSocketLogHandler`: Forwards all root logger output to Electron UI via `UIBridge.emit_log()`
