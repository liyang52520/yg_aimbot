# YG Aimbot

Written by Kimi GLM, Seed GLM, DouBao Seed GLM, MiniMax GLM and YG people;

### capture

Only support mss capture;

### move

Only support makcu;

### env

- python 3.10
- cuda 12.8
- tensorrt 10.9

### install

```shell
pip install --force-reinstall -r requirements.txt
pip install --force-reinstall torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install .\tensorrt-10.9.0.34-cp310-none-win_amd64.whl
pip install --force-reinstall install onnx==1.12.0
pip install --force-reinstall install onnxslim==0.1.71
pip install --force-reinstall install numpy==1.26.4
```

### dlls

- I rebuild the onnxruntime.dll, it can export onnx model to root dir of C:/ after load, you can use it to export model
  for encrypted application which use onnxruntime.dll
- 我重新编译了onnxruntime.dll，它可以在加载onnx模型时导出onnx模型到c盘根目录，你可以用它来导出加密的应用程序，比如：天机ai

### models

- Apex model (YOLOv8s) provided by YG, you can see it in `models/`

### start

```shell
python run.py
```