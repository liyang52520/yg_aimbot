# YG Aimbot / YG 辅助瞄准

<div align="center">
  <img src="ui/resources/aim.svg" alt="YG Aimbot Logo" width="85"/>

  <p>A high-performance, AI-powered aimbot for FPS with advanced features and an intuitive GUI</p>
  <p>一款高性能、AI驱动的FPS自瞄工具，具有先进功能和直观的GUI界面</p>

  <div>
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10"/>
    <img src="https://img.shields.io/badge/CUDA-12.8-green.svg" alt="CUDA 12.8"/>
    <img src="https://img.shields.io/badge/TensorRT-10.9-orange.svg" alt="TensorRT 10.9"/>
    <img src="https://img.shields.io/badge/Ultralytics-darkblue.svg" alt="Ultralytics"/>
  </div>
</div>

## 📋 Table of Contents / 目录

- [Features](#features) / [功能](#features)
- [Requirements](#requirements) / [要求](#requirements)
- [Installation](#installation) / [安装](#installation)
- [Quick Start](#quick-start) / [快速开始](#quick-start)
- [Configuration](#configuration) / [配置](#configuration)
- [Usage](#usage) / [使用](#usage)
- [Technical Architecture](#technical-architecture) / [技术架构](#technical-architecture)
- [Performance](#performance) / [性能](#performance)
- [FAQ](#faq) / [常见问题](#faq)
- [Disclaimer](#disclaimer) / [免责声明](#disclaimer)

## ✨ Features / 功能

- **Advanced AI Detection**: Uses YOLOv8s model optimized for Apex Legends to detect enemies with high accuracy
  **先进的AI检测**：使用针对Apex Legends优化的YOLOv8s模型，高精度检测敌人
- **Asynchronous Inference**: Double-buffered design for smooth performance
  **异步推理**：双缓冲设计，实现流畅性能
- **Precise Aiming**: Customizable smoothness, prediction, and tremor effects
  **精确瞄准**：可自定义平滑度、预测和抖动效果
- **Intuitive GUI**: Easy-to-use interface for configuration
  **直观的GUI**：易于使用的配置界面
- **Hotkey Support**: Configurable hotkeys with both hold and toggle modes
  **热键支持**：可配置热键，支持按住和切换模式
- **Real-time Visualization**: Live detection feedback
  **实时可视化**：实时检测反馈

## 🛠️ Requirements / 要求

| Component   | Version | 组件          | 版本   |
|-------------|---------|-------------|------|
| Python      | 3.10    | Python      | 3.10 |
| CUDA        | 12.8    | CUDA        | 12.8 |
| TensorRT    | 10.9    | TensorRT    | 10.9 |
| PyQt6       | Latest  | PyQt6       | 最新版  |
| Ultralytics | Latest  | Ultralytics | 最新版  |
| mss         | Latest  | mss         | 最新版  |
| makcu       | Latest  | makcu       | 最新版  |

## 📦 Installation / 安装

### Step 1: Clone the repository / 步骤1：克隆仓库

```bash
git clone https://github.com/yourusername/yg-aimbot.git
cd yg-aimbot
```

### Step 2: Install dependencies / 步骤2：安装依赖

```bash
pip install --force-reinstall -r requirements.txt
pip install --force-reinstall torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

### Step 3: Install TensorRT / 步骤3：安装TensorRT

```bash
pip install ./tensorrt-10.9.0.34-cp310-none-win_amd64.whl
```

### Step 4: Install additional dependencies / 步骤4：安装额外依赖

```bash
pip install --force-reinstall numpy==1.26.4
```

## 🚀 Quick Start / 快速开始

1. **Start the application**: / **启动应用**：

```bash
python run.py
```

2. **Configure settings** in the GUI: / **在GUI中配置设置**：
    - Adjust capture settings (resolution, FPS) / 调整捕获设置（分辨率、FPS）
    - Set AI parameters (confidence threshold) / 设置AI参数（置信度阈值）
    - Configure aim settings (smoothness, prediction) / 配置瞄准设置（平滑度、预测）
    - Set hotkeys for activation / 设置激活热键

3. **Launch Apex Legends** and start playing with your configured settings! / **启动Apex Legends**并使用配置的设置开始游戏！

## ⚙️ Configuration / 配置

The application uses a config.ini file for persistent settings. You can also configure everything through the GUI.
应用使用config.ini文件存储持久设置。您也可以通过GUI配置所有内容。

### Key Configuration Sections / 关键配置部分

- **[capture]**: Screen capture settings (resolution, FPS, circle mask) / **[capture]**：屏幕捕获设置（分辨率、FPS、圆形掩码）
- **[ai]**: AI detection settings (model, confidence threshold, device) / **[ai]**：AI检测设置（模型、置信度阈值、设备）
- **[aim]**: Aiming settings (hotkeys, mode, smoothness, prediction) / **[aim]**：瞄准设置（热键、模式、平滑度、预测）
- **[mouse]**: Mouse settings (DPI, sensitivity, FOV) / **[mouse]**：鼠标设置（DPI、灵敏度、FOV）

### Example Configuration / 配置示例

```ini
[capture]
window_width = 320
window_height = 320
fps = 60
circle = false
ai_debug = false

[ai]
model_name = YOLOv8s_apex_teammate_enemy.engine
device = 0
conf = 0.2

[aim]
hotkeys = X1MouseButton,X2MouseButton
mode = hold
auto = false
body_x_offset = 0.1
body_y_offset = 0.1

[mouse]
dpi = 1100
sensitivity = 3.0
fov_width = 40
fov_height = 40
```

## 🎮 Usage / 使用

### Hotkey Modes / 热键模式

- **Hold Mode**: Press and hold the hotkey to activate aimbot / **按住模式**：按住热键激活自瞄
- **Toggle Mode**: Press the hotkey once to enable, press again to disable / **切换模式**：按一次热键启用，再按一次禁用
- **Auto Mode**: Aimbot is always active / **自动模式**：自瞄始终处于活动状态

### GUI Controls / GUI控件

- **AI Config Tab**: Adjust model settings and view detection results / **AI配置选项卡**：调整模型设置并查看检测结果
- **Aim Config Tab**: Configure aiming behavior and hotkeys / **瞄准配置选项卡**：配置瞄准行为和热键
- **Log Window**: View real-time application logs / **日志窗口**：查看实时应用日志
- **Performance Metrics**: Monitor FPS and inference times / **性能指标**：监控FPS和推理时间

## 🏗️ Technical Architecture / 技术架构

### Core Services / 核心服务

1. **Config Service**:
    - Manages application settings
    - Provides real-time configuration updates
      **配置服务**：
    - 管理应用设置
    - 提供实时配置更新
2. **Capture Service**:
    - Uses mss for high-performance screen capture
    - Supports configurable capture area
    - Implements circular mask option
      **捕获服务**：
    - 使用mss进行高性能屏幕捕获
    - 支持可配置的捕获区域
    - 实现圆形掩码选项
3. **Inference Service**:
    - Asynchronous double-buffered design
    - YOLOv8 model with TensorRT optimization
    - Adaptive FPS for optimal performance
      **推理服务**：
    - 异步双缓冲设计
    - 带有TensorRT优化的YOLOv8模型
    - 自适应FPS以获得最佳性能

4. **Tracker Service**:
    - Tracks detected targets
    - Improves target selection accuracy
      **跟踪服务**：
    - 跟踪检测到的目标
    - 提高目标选择准确性
5. **Aim Service**:
    - Precise mouse control using makcu
    - Smooth aiming with customizable parameters
    - Target prediction and tremor effects
      **瞄准服务**：
    - 使用makcu进行精确鼠标控制
    - 可自定义参数的平滑瞄准
    - 目标预测和抖动效果

### Flow Diagram / 流程图

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ Capture Service │ ──> │ Inference Service │ ──> │  Tracker Service │ ──> │  Aim Service   │ ──> │ Mouse Control   │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## ❓ FAQ / 常见问题

### Q: Is this detectable by anti-cheat systems?

A: While we've implemented various techniques to minimize detection risk, using any third-party software in online games
carries inherent risks. Use at your own discretion.

### Q：这会被反作弊系统检测到吗？

A：虽然我们已经实施了各种技术来最小化检测风险，但在在线游戏中使用任何第三方软件都存在固有风险。请自行决定是否使用。

### Q: Can I use this with other games?

A: The current model is optimized for Apex Legends. You would need to train a custom model for other games.

### Q：我可以在其他游戏中使用吗？

A：当前模型针对Apex Legends进行了优化。您需要为其他游戏训练自定义模型。

### Q: How do I improve accuracy?

A: Adjust the confidence threshold in the AI settings and fine-tune the aim parameters for your playstyle.

### Q：如何提高准确性？

A：调整AI设置中的置信度阈值，并根据您的游戏风格微调瞄准参数。

### Q: What if I get a "model not found" error?

A: Ensure the model file is present in the `data/` directory and that the model name in config matches the actual
filename.

### Q：如果出现"model not found"错误怎么办？

A：确保模型文件存在于`data/`目录中，并且配置中的模型名称与实际文件名匹配。

## 📝 Disclaimer / 免责声明

This software is provided for educational purposes only. The developers are not responsible for any consequences
resulting from the use of this software. Using third-party software to gain an advantage in online games may violate the
game's terms of service and result in account bans. Use at your own risk.
本软件仅用于教育目的。开发者对使用本软件产生的任何后果不承担责任。使用第三方软件在在线游戏中获得优势可能违反游戏的服务条款，并导致账号被封禁。请自行承担风险。

## 🤝 Contributing / 贡献

Contributions are welcome! Please feel free to submit a Pull Request.
欢迎贡献！请随时提交Pull Request。

## 📄 License / 许可证

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 👥 Authors / 作者

- Written by Kimi GLM, DouBao Seed GLM, MiniMax GLM and YG team
- 由Kimi GLM、DouBao Seed GLM、MiniMax GLM和YG团队编写

---

<div align="center">
  <p>Made with ❤️ for the gaming community</p>
  <p>为游戏社区精心打造 ❤️</p>
</div>