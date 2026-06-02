# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

- **Python 3.10** conda environment named `yg_aimbot` — activate with `conda activate yg_aimbot` before running anything
- Python dependencies: `pip install -r requirements.txt`
- Electron UI dependencies: `cd electron_ui && npm install`
- Models go in the `data/` directory

## Commands

- **Run application**: `python run.py` (launches both core + Electron UI)
- **Run Electron UI standalone**: `cd electron_ui && npm start` (for frontend-only development, requires Python core running separately)
- **Install Python deps**: `pip install --force-reinstall -r requirements.txt`
- **Tests**: No test files committed yet (only `.pyc` artifacts exist in `core/testing/`)

## Project Architecture

### Data Flow Pipeline

```
CaptureService → InferenceService → TrackerService → AimService → MouseControlService
     ↑                                                                              |
     └────────────────── WebSocket (port 8765) → Electron UI ──────────────────────┘
```

### Core Modules (`core/`)

| File | Purpose |
|------|---------|
| `aimbot.py` | Application main class — owns the async main loop, hotkey state machine, wires all services together |
| `ui_bridge.py` | Signal bridge (replaces PyQt signals) — forwards events to both legacy listeners and WebSocket |
| `websocket_server.py` | WebSocket server on port 8765 — handles config commands, pushes FPS/video/detections/logs to Electron |
| `buttons.py` | Virtual key code mapping (Win32) for all supported mouse buttons, keyboard keys, and media keys |
| `logger.py` | Colored console logging + WebSocket log forwarding via `WebSocketLogHandler` |

### Services (`core/services/`)

All services are **singletons** instantiated at module level (e.g., `config_service = ConfigService()`).

**`config_service.py`** — Thread-safe INI-based config with default fallback, `Lock`-protected reads/writes, and callback registration for live config changes.

**`capture_service.py`** — Screen capture via `mss`. Captures a centered region of the primary monitor. Supports circular mask. Generates monotonic frame IDs using a snowflake-like timestamp+counter scheme.

**`inference_service.py`** — Async double-buffered inference. Main thread calls `submit_frame()`, worker thread runs `YOLO()` model, callback thread processes results. Supports `.pt`, `.engine`, `.onnx` models. Uses `supervision.Detections` for output.

**`tracker_service.py`** — Target selection & continuity tracking. Picks the best detection by distance-to-center and class filter (`target_cls`). Maintains tracking continuity with a velocity-based `TargetPredictor` that fills gaps during brief detection misses (up to `max_miss_time`).

**`aim_service.py`** — Computes mouse movement deltas. Pipeline: body offset → model-space centering → velocity prediction → smoothing → tremor → DPI/sensitivity scaling → `makcu` hardware mouse move. Runs a dedicated `_move_worker` thread with a bounded queue.

### UI (`electron_ui/`)

- **Vue.js 2** SPA (CDN-loaded, no build step) in `app.js`
- **Element UI** component library (CDN)
- Communication via WebSocket to `ws://127.0.0.1:8765`
- Config sections map 1:1 between `config.ini` and the UI `config` data object
- Hotkey multiselect, body offset drag canvas, video stream with detection overlay

### Config (`config.ini`)

Four sections: `[AI]`, `[Capture]`, `[Aim]`, `[Mouse]`. Config is loaded on startup and can be changed at runtime via WebSocket or by editing `config.ini` directly (requires restart).

### Key Design Decisions

- **No PyQt** — The original PyQt UI was replaced with Electron. `UIBridge` provides the same signal interface without PyQt dependency.
- **Threading model**: Capture runs in its own thread, inference in its own thread, mouse movement in its own thread, WebSocket in its own thread with an asyncio event loop. Main loop is async and coordinates via thread-safe queues and locks.
- **Frame ID snowflake scheme**: `(current_second << 20) | counter` — ensures monotonic IDs for deduplication.
- **Target prediction**: `TrackerService` uses a velocity-based linear extrapolator to maintain aim during brief detection gaps (up to 150ms), with configurable `max_miss_time` and `max_miss_distance`.
