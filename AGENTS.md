# yg_aimbot - AI Aimbot Project

## 项目概述

YG Aimbot 是一款基于 YOLOv8 的 Apex Legends AI 自瞄工具，采用双缓冲异步推理架构实现高性能目标检测和精确瞄准控制。

## 运行环境

1. **Python 环境**: 必须使用 conda 虚拟环境 `yg_aimbot`
2. **Conda 启动命令**: `conda activate yg_aimbot`
3. **运行项目命令**: `python run.py`

## 项目架构

```
yg_aimbot/
├── core/                          # 核心模块
│   ├── aimbot.py                  # 自瞄应用主类，主循环和热键管理
│   ├── buttons.py                 # 热键按键码定义
│   ├── logger.py                  # 日志配置
│   ├── ui_bridge.py               # UI桥接模块（替代PyQt信号）
│   ├── websocket_server.py        # WebSocket服务器（Electron通信）
│   └── services/                  # 核心服务
│       ├── config_service.py      # 配置管理服务（单例）
│       ├── capture_service.py     # 屏幕捕获服务（mss/bettercam）
│       ├── inference_service.py   # 异步推理服务（YOLOv8）
│       ├── tracker_service.py     # 目标跟踪服务
│       └── aim_service.py         # 瞄准控制服务（makcu）
├── electron_ui/                   # Electron UI 模块
│   ├── main.js                    # Electron主进程
│   ├── app.js                     # Vue前端应用
│   ├── index.html                 # 主页面
│   ├── styles.css                 # 样式文件
│   ├── config_bridge.py           # 配置桥接脚本（Python端）
│   └── package.json               # Node依赖
├── data/                          # 模型文件目录
│   ├── YOLOv8s_apex_teammate_enemy.engine  # TensorRT引擎模型
│   ├── YOLOv8s_apex_teammate_enemy.pt      # PyTorch模型
│   └── onnxruntime.dll            # ONNX运行时库
├── config.ini                     # 配置文件
├── run.py                         # 入口文件
└── requirements.txt               # Python依赖
```

## 核心服务详解

### 1. ConfigService (config_service.py)

配置管理服务，采用单例模式，线程安全。

**配置结构**:
```python
DEFAULT_CONFIG = {
    'capture': {           # 屏幕捕获配置
        'window_width': 320,
        'window_height': 320,
        'fps': 60,
        'circle': True,   # 圆形掩码
        'ai_debug': False  # AI调试模式（视频预览）
    },
    'ai': {               # AI检测配置
        'model_name': 'YOLOv8s_apex_teammate_enemy.engine',
        'conf': 0.2,      # 置信度阈值
        'device': '0'     # GPU设备ID
    },
    'aim': {              # 瞄准配置
        'auto': False,    # 自动瞄准开关
        'mode': 'hold',   # hold/toggle模式
        'target_cls': 1.0,# 目标类别（0=队友，1=敌人）
        'body_x_offset': 0.1,   # 身体X偏移
        'body_y_offset': 0.1,   # 身体Y偏移
        'hotkeys': 'X1MouseButton,X2MouseButton',
        'max_target_distance': 90  # 最大目标距离（像素）
    },
    'mouse': {            # 鼠标配置
        'move': 'makcu',  # 鼠标控制方式
        'dpi': 1100,      # DPI设置
        'sensitivity': 3.0,       # 灵敏度
        'fov_width': 40,          # 水平视野
        'fov_height': 40          # 垂直视野
    }
}
```

**关键方法**:
- `get(section, key, default)` - 获取配置值
- `set(section, key, value)` - 设置配置值
- `save()` - 保存配置到 config.ini
- `get_section(section)` - 获取整个配置节
- `update_section(section, updates, notify)` - 批量更新配置节
- `register_callback(callback)` / `unregister_callback(callback)` - 注册/注销配置变更回调

### 2. CaptureService (capture_service.py)

屏幕捕获服务，支持 mss 和 bettercam 两种后端。

**关键方法**:
- `start()` / `stop()` - 启动/停止捕获
- `get_frame()` - 获取当前帧和帧ID（返回: frame, frame_id）
- `get_resolution()` - 获取捕获分辨率

**帧ID生成**: 使用基于时间的雪花算法，确保帧ID唯一且递增。

### 3. InferenceService (inference_service.py)

异步推理服务，双缓冲设计实现高性能目标检测。

**关键组件**:
- `InferenceResult` - 推理结果数据类
- `_inference_worker` - 推理工作线程
- `_result_worker` - 结果处理线程

**关键方法**:
- `load()` - 加载 YOLOv8 模型
- `start()` / `stop()` - 启动/停止服务
- `submit_frame(frame, frame_id)` - 提交帧进行推理
- `get_latest_result()` - 获取最新推理结果
- `get_input_size()` - 获取模型输入大小

**模型支持**: `.pt` (PyTorch), `.engine` (TensorRT), `.onnx` 格式

### 4. TrackerService (tracker_service.py)

目标跟踪服务，基于距离和类别选择最佳目标。

**目标选择逻辑**:
1. 计算所有检测目标到中心点的距离
2. 优先跟踪已跟踪目标（距离阈值内保持连续性）
3. 否则选择距离中心最近的目标
4. 应用目标类别过滤（target_cls）

**关键方法**:
- `update(detections, frame_id)` - 更新检测结果
- `get_best_target()` - 获取当前跟踪目标
- `reset()` - 重置跟踪状态

### 5. AimService (aim_service.py)

瞄准控制服务，处理鼠标移动计算和执行。

**移动计算流程**:
1. 应用身体偏移量（body_x_offset, body_y_offset）调整目标位置
2. 计算到模型中心的偏移
3. 基于历史数据计算速度进行预测
4. 应用平滑因子减少抖动
5. 转换为像素移动
6. 添加随机抖动效果
7. 执行鼠标移动（通过 makcu）

**关键方法**:
- `process_target(x, y, w, h)` - 处理目标并执行瞄准
- `set_model_input_size(size)` - 设置模型输入大小
- `set_config()` - 应用新配置

## UI 架构（Electron + Vue）

### 通信架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Electron UI                              │
│  ┌─────────────────┐        ┌──────────────────────────┐   │
│  │   Renderer      │        │        Main Process      │   │
│  │   (Vue + app.js)│◄──────►│        (main.js)         │   │
│  │                 │  IPC   │                          │   │
│  └────────┬────────┘        └──────────┬───────────────┘   │
│           │                            │                    │
│           │ WebSocket                  │ stdio pipe         │
│           │ (Port 8765)                │                    │
└───────────┼────────────────────────────┼────────────────────┘
            │                            ▼
            │                   ┌─────────────────┐
            │                   │ config_bridge.py│
            │                   │ (配置桥接脚本)   │
            │                   └─────────────────┘
            │                            │
            ▼                            ▼
   ┌─────────────────┐         ┌─────────────────┐
   │ websocket_server│         │ config_service  │
   │   (WebSocket)   │         │   (配置管理)     │
   └─────────────────┘         └─────────────────┘
            │                            │
            └────────────┬───────────────┘
                         │
                         ▼
              ┌─────────────────┐
              │   ui_bridge     │
              │  (信号桥接)      │
              └─────────────────┘
```

### WebSocket 通信协议

**消息类型**:

| 类型 | 方向 | 说明 |
|------|------|------|
| `config` | Server → Client | 发送当前完整配置 |
| `save-config` | Client → Server | 保存配置到文件 |
| `apply-config` | Client → Server | 应用配置到内存 |
| `scan-models` | Client → Server | 扫描模型文件 |
| `fps` | Server → Client | 实时帧率数据 |
| `video-frame` | Server → Client | 视频帧（Base64 JPEG） |
| `detections` | Server → Client | 检测结果数据 |
| `log` | Server → Client | 日志消息 |

### 前端功能（app.js）

**标签页**:
- **AI 配置**: 模型选择、置信度、设备ID
- **捕获配置**: 窗口大小、帧率、圆形掩码、AI调试
- **瞄准配置**: 模式、目标类别、身体偏移可视化、热键
- **鼠标配置**: DPI、灵敏度、视野范围

**可视化功能**:
- 实时视频流显示（带检测框）
- 身体偏移瞄准点可视化（可拖拽调整）
- FPS 实时监控
- 日志显示

## 数据流

```
CaptureService.get_frame()
        ↓
Aimbot._main_loop() 接收帧
        ↓
WebSocketServer.emit_video_frame() 发送视频帧到UI
        ↓
inference_service.submit_frame() 提交推理
        ↓
inference_service._inference_worker 执行推理
        ↓
tracker_service.update() 更新目标
        ↓
aim_service.process_target() 计算移动
        ↓
MouseControlService.move() 执行鼠标移动
```

## 热键定义 (buttons.py)

支持的按键包括:
- **鼠标**: `LeftMouseButton`, `RightMouseButton`, `MiddleMouseButton`, `X1MouseButton`, `X2MouseButton`
- **键盘**: `A-Z`, `F1-F12`, `LeftShift`, `RightShift`, `LeftControl`, `RightControl`, `LeftAlt`, `RightAlt`, `Space`, `Tab`, `Enter` 等

## GUI 配置映射

| config.ini Section | config.ini Key | UI 字段 | 类型 |
|-------------------|----------------|---------|------|
| AI | model_name | ai.model_name | Select |
| AI | conf | ai.conf | Slider (0-1) |
| AI | device | ai.device | Number |
| Capture | window_width | capture.window_width | Number |
| Capture | window_height | capture.window_height | Number |
| Capture | fps | capture.fps | Number |
| Capture | circle | capture.circle | Switch |
| Capture | ai_debug | videoEnabled | Switch |
| Aim | auto | aim.auto | Switch |
| Aim | mode | aim.mode | Select (hold/toggle) |
| Aim | target_cls | aim.target_cls | Select (0=队友,1=敌人) |
| Aim | body_x_offset | aim.body_x_offset | Slider (-1~1) |
| Aim | body_y_offset | aim.body_y_offset | Slider (-1~1) |
| Aim | hotkeys | selectedHotkeys | Multi-select |
| Aim | max_target_distance | aim.max_target_distance | Number |
| Mouse | move | mouse.move | Select |
| Mouse | dpi | mouse.dpi | Number |
| Mouse | sensitivity | mouse.sensitivity | Number |
| Mouse | fov_width | mouse.fov_width | Number |
| Mouse | fov_height | mouse.fov_height | Number |

## 依赖

### Python 依赖
- **Python 3.10**
- **ultralytics==8.4.15** - YOLOv8 目标检测
- **opencv-python==4.6.0.66** - 图像处理
- **supervision==0.27.0.post1** - 检测结果后处理
- **onnxruntime-gpu==1.23.2** - ONNX推理
- **makcu==2.3.1** - 鼠标控制
- **bettercam==1.0.0** - 屏幕捕获
- **keyboard==0.13.5** - 全局热键
- **websockets==12.0** - WebSocket通信
- **pywin32==311** - Windows API

### Node.js 依赖
- **electron** - 桌面应用框架
- **vue** (CDN) - 前端框架
- **element-ui** (CDN) - UI组件库

## 开发指南

### 添加新配置项

1. 在 `ConfigService.DEFAULT_CONFIG` 中添加默认值
2. 在 `config.ini` 中添加对应项
3. 在 `electron_ui/app.js` 的 `data()` 中添加配置字段
4. 在 `handleConfigReceived()` 方法中处理配置加载
5. 在对应标签页添加 UI 控件

### 添加新服务

1. 创建服务类，实现 `start()`, `stop()` 方法
2. 在 `core/aimbot.py` 中初始化服务
3. 在主循环中调用服务方法
4. 如有需要，在 `ui_bridge.py` 中添加信号

### 修改瞄准算法

瞄准计算逻辑在 `AimService._calculate_movement()` 方法中，包括:
- 偏移调整（应用 body_x_offset, body_y_offset）
- 速度预测（基于历史位置计算）
- 平滑处理（smooth_factor）
- 抖动效果（tremor_amount）

### Electron UI 开发

1. 前端代码在 `electron_ui/app.js`（Vue）
2. 样式在 `electron_ui/styles.css`
3. 主进程在 `electron_ui/main.js`
4. WebSocket 服务器在 `core/websocket_server.py`

**启动 Electron 开发模式**:
```bash
cd electron_ui
npm install
npm start
```

## 注意事项

1. 所有服务均使用单例模式，通过 `*_service = ServiceClass()` 创建全局实例
2. 配置服务线程安全，使用 `Lock` 保护
3. WebSocket 服务器在 `run.py` 中启动，端口 8765
4. 配置变更通过 WebSocket 实时同步到 UI
5. 视频帧通过 Base64 JPEG 编码传输，限制 30fps
6. 异步推理服务使用双缓冲，主线程提交帧，工作线程执行推理
7. Electron UI 需要 Node.js 环境，Python 核心需要 Conda 环境
