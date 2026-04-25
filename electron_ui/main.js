const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonBridge = null;

// 启动 Python 桥接进程
function startPythonBridge() {
  const pythonScript = path.join(__dirname, 'config_bridge.py');
  
  // 使用 conda 环境的 Python
  const pythonExe = 'python'; // 假设 python 在 PATH 中
  
  pythonBridge = spawn(pythonExe, [pythonScript], {
    stdio: ['pipe', 'pipe', 'pipe']
  });
  
  pythonBridge.stderr.on('data', (data) => {
    console.error(`Python Error: ${data}`);
  });
  
  pythonBridge.on('close', (code) => {
    console.log(`Python bridge exited with code ${code}`);
    pythonBridge = null;
  });
}

// 向 Python 桥接发送命令
function sendCommand(command, data = {}) {
  return new Promise((resolve, reject) => {
    if (!pythonBridge) {
      reject(new Error('Python bridge not available'));
      return;
    }
    
    const request = JSON.stringify({ command, data });
    
    // 设置一次性数据监听器
    const onData = (data) => {
      try {
        const response = JSON.parse(data.toString().trim());
        pythonBridge.stdout.off('data', onData);
        resolve(response);
      } catch (e) {
        // 可能收到不完整数据，忽略
      }
    };
    
    pythonBridge.stdout.on('data', onData);
    
    // 发送命令
    pythonBridge.stdin.write(request + '\n');
    
    // 超时处理
    setTimeout(() => {
      pythonBridge.stdout.off('data', onData);
      reject(new Error('Command timeout'));
    }, 5000);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1100,
    height: 750,
    minWidth: 900,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      webSecurity: false
    },
    titleBarStyle: 'default',
    show: true,
    autoHideMenuBar: true
  });
  
  // 移除菜单栏
  mainWindow.setMenu(null);

  // 加载错误处理
  mainWindow.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('页面加载失败:', errorCode, errorDescription);
  });

  mainWindow.webContents.on('console-message', (event, level, message, line, sourceId) => {
    const levels = ['debug', 'log', 'warn', 'error'];
    console.log(`[Console ${levels[level]}]`, message);
  });

  mainWindow.loadFile('index.html').catch(err => {
    console.error('加载文件失败:', err);
  });

  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    mainWindow.focus();
  });

  // 开发者工具 - 按 F12 手动打开
  // mainWindow.webContents.openDevTools();

  mainWindow.on('closed', () => {
    mainWindow = null;
    if (pythonBridge) {
      pythonBridge.kill();
    }
  });
}

// IPC 处理器
ipcMain.handle('get-config', async () => {
  try {
    const result = await sendCommand('get-config');
    return result;
  } catch (error) {
    console.error('获取配置失败:', error);
    return null;
  }
});

ipcMain.handle('save-config', async (event, config) => {
  try {
    const result = await sendCommand('save-config', config);
    return result;
  } catch (error) {
    console.error('保存配置失败:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('apply-config', async (event, config) => {
  try {
    const result = await sendCommand('apply-config', config);
    return result;
  } catch (error) {
    console.error('应用配置失败:', error);
    return { success: false, error: error.message };
  }
});

ipcMain.handle('scan-models', async () => {
  try {
    const result = await sendCommand('scan-models');
    return result;
  } catch (error) {
    console.error('扫描模型失败:', error);
    return ['YOLOv8s_apex_teammate_enemy.engine'];
  }
});

app.whenReady().then(() => {
  startPythonBridge();
  createWindow();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
