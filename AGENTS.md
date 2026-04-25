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
│   └── services/                  # 核心服务
│       ├── config_service.py      # 配置管理服务（单例）
│       ├── capture_service.py     # 屏幕捕获服务（mss）
│       ├── inference_service.py   # 异步推理服务（YOLOv8）
│       ├── tracker_service.py      # 目标跟踪服务
│       └── aim_service.py          # 瞄准控制服务（makcu）
├── ui/                            # PyQt6 GUI 模块
│   ├── main_window.py             # 主窗口，协调各组件
│   ├── config_manager.py          # UI配置管理器
│   ├── signals.py                  # Qt信号定义
│   ├── styles/                     # 样式定义
│   ├── tabs/                       # 配置标签页
│   │   ├── ai_config_tab.py       # AI和捕获配置
│   │   └── aim_config_tab.py      # 瞄准和鼠标配置
│   └── widgets/                    # 自定义组件
├── data/                           # 模型文件目录
│   ├── YOLOv8s_apex_teammate_enemy.engine  # TensorRT.engine模型
│   └── YOLOv8s_apex_teammate_enemy.pt      # PyTorch模型
├── config.ini                      # 配置文件
└── run.py                          # 入口文件
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
        'ai_debug': False
    },
    'ai': {               # AI检测配置
        'model_name': 'YOLOv8s_apex_teammate_enemy.engine',
        'conf': 0.2,      # 置信度阈值
        'device': '0'     # GPU设备ID
    },
    'aim': {              # 瞄准配置
        'auto': False,    # 自动瞄准
        'mode': 'hold',   # hold/toggle
        'target_cls': 1.0,  # 目标类别
        'body_x_offset': 0.1,
        'body_y_offset': 0.1,
        'hotkeys': 'X1MouseButton,X2MouseButton',
        'max_target_distance': 90
    },
    'mouse': {            # 鼠标配置
        'move': 'makcu',
        'dpi': 1100,
        'sensitivity': 3.0,
        'fov_width': 40,
        'fov_height': 40
    }
}
```

**关键方法**:
- `get(section, key, default)` - 获取配置值
- `set(section, key, value)` - 设置配置值
- `save()` - 保存配置到 config.ini
- `register_callback(callback)` - 注册配置变更回调

### 2. CaptureService (capture_service.py)

屏幕捕获服务，基于 mss 库，支持可配置帧率和圆形掩码。

**关键方法**:
- `start()` / `stop()` - 启动/停止捕获
- `get_frame()` - 获取当前帧和帧ID
- `get_resolution()` - 获取捕获分辨率

**帧ID生成**: 使用基于时间的雪花算法，确保帧ID唯一且递增。

### 3. InferenceService (inference_service.py)

异步推理服务，双缓冲设计。

**关键组件**:
- `InferenceResult` - 推理结果数据类
- `_inference_worker` - 推理工作线程
- `_result_worker` - 结果处理线程

**关键方法**:
- `load()` - 加载 YOLOv8 模型
- `start()` / `stop()` - 启动/停止服务
- `submit_frame(frame, frame_id)` - 提交帧进行推理
- `get_latest_result()` - 获取最新推理结果

**模型支持**: `.pt` (PyTorch), `.engine` (TensorRT), `.onnx` 格式

### 4. TrackerService (tracker_service.py)

目标跟踪服务，基于距离和类别选择最佳目标。

**目标选择逻辑**:
1. 计算所有检测目标到中心点的距离
2. 优先跟踪已跟踪目标（距离阈值内）
3. 否则选择距离中心最近的目标
4. 应用目标类别过滤

**关键方法**:
- `update(detections, frame_id)` - 更新检测结果
- `get_best_target()` - 获取当前跟踪目标
- `reset()` - 重置跟踪状态

### 5. AimService (aim_service.py)

瞄准控制服务，处理鼠标移动计算和执行。

**MouseConfig 配置**:
- `dpi`, `sensitivity` - 鼠标参数
- `fov_width`, `fov_height` - 视野范围
- `smooth_factor` - 平滑因子 (0-1)
- `max_move`, `min_move` - 移动限制
- `tremor_amount` - 抖动幅度
- `prediction_time` - 预测时间

**移动计算流程**:
1. 应用身体偏移量调整目标位置
2. 计算到模型中心的偏移
3. 基于历史数据计算速度进行预测
4. 应用平滑因子
5. 转换为像素移动
6. 添加抖动效果
7. 执行鼠标移动

**关键方法**:
- `process_target(x, y, w, h)` - 处理目标
- `set_model_input_size(size)` - 设置模型输入大小

## 数据流

```
CaptureService.get_frame()
        ↓
Aimbot._main_loop() 接收帧
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

## 信号系统 (signals.py)

**LogSignal**:
- `log` - 日志信号

**ImageSignal**:
- `image` - 原始帧图像
- `capture_fps` - 捕获帧率
- `predict_fps` - 推理帧率
- `detection_result` - 检测结果

## 热键定义 (buttons.py)

支持的按键包括:
- 鼠标: `LeftMouseButton`, `RightMouseButton`, `MiddleMouseButton`, `X1MouseButton`, `X2MouseButton`
- 键盘: `A-Z`, `F1-F12`, `LeftShift`, `RightShift`, `LeftControl`, `RightControl`, 等

## GUI 配置映射

| config.ini Section | config.ini Key | UI Component | 类型 |
|-------------------|----------------|--------------|------|
| AI | model_name | ai_model_name | QComboBox |
| AI | conf | ai_conf | QDoubleSpinBox |
| AI | device | ai_device | QSpinBox |
| Capture | window_width | capture_window_width | QSpinBox |
| Capture | window_height | capture_window_height | QSpinBox |
| Capture | fps | capture_fps | QSpinBox |
| Capture | circle | capture_circle | QCheckBox |
| Capture | ai_debug | capture_ai_debug | QCheckBox |
| Aim | auto | auto | QCheckBox |
| Aim | mode | aim_mode | QComboBox |
| Aim | target_cls | target_cls | QComboBox |
| Aim | body_x_offset | body_x_offset | QDoubleSpinBox |
| Aim | body_y_offset | body_y_offset | QDoubleSpinBox |
| Aim | hotkeys | hotkeys | MultiSelectDropDown |
| Aim | max_target_distance | max_target_distance | QSpinBox |
| Mouse | move | mouse_move | QComboBox |
| Mouse | dpi | mouse_dpi | QSpinBox |
| Mouse | sensitivity | mouse_sensitivity | QDoubleSpinBox |
| Mouse | fov_width | mouse_fov_width | QSpinBox |
| Mouse | fov_height | mouse_fov_height | QSpinBox |

## 依赖

- **Python 3.10**
- **PyTorch 2.9.1** + CUDA 12.8
- **TensorRT 10.9** (用于 .engine 模型)
- **Ultralytics YOLO 8.4.15**
- **PyQt6 6.10.2** - GUI框架
- **mss** - 屏幕捕获
- **makcu** - 鼠标控制
- **opencv-python 4.6.0.66**
- **supervision** - 检测结果后处理

## 开发指南

### 添加新配置项

1. 在 `ConfigService.DEFAULT_CONFIG` 中添加默认值
2. 在 `config.ini` 中添加对应项
3. 在 UI 对应 Tab 的 `_create_*_config_section` 方法中添加控件
4. 在 `MainWindow._get_ui_components()` 中注册控件
5. 在 `ConfigManager` 中处理新配置项

### 添加新服务

1. 创建服务类，实现 `start()`, `stop()` 方法
2. 在 `core.aimbot.Aimbot` 中初始化服务
3. 在主循环中调用服务方法

### 修改瞄准算法

瞄准计算逻辑在 `AimService._calculate_movement()` 方法中，包括:
- 偏移调整
- 速度预测
- 平滑处理
- 抖动效果

## 注意事项

1. 所有服务均使用单例模式，通过 `*_service = ServiceClass()` 创建全局实例
2. 配置服务线程安全，使用 `Lock` 保护
3. GUI 配置变更自动应用到内存，如需保存到文件需点击"保存配置"按钮
4. 异步推理服务使用双缓冲，主线程提交帧，工作线程执行推理
