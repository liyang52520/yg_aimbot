new Vue({
  el: '#app',
  data() {
    return {
      activeTab: 'ai',
      saving: false,
      videoEnabled: false,
      videoFrame: '',
      captureFps: 0,
      predictFps: 0,
      detectionCount: 0,
      detections: [],
      logs: [],
      modelList: ['YOLOv8s_apex_teammate_enemy.engine'],
      selectedHotkeys: [],
      isDragging: false,
      
      ws: null,
      wsConnected: false,
      wsReconnectTimer: null,
      
      config: {
        ai: {
          model_name: 'YOLOv8s_apex_teammate_enemy.engine',
          conf: 0.2,
          device: 0
        },
        capture: {
          window_width: 320,
          window_height: 320,
          fps: 60,
          circle: true,
          ai_debug: false
        },
        aim: {
          auto: false,
          mode: 'hold',
          target_cls: 1,
          body_x_offset: 0.1,
          body_y_offset: 0.1,
          hotkeys: 'X1MouseButton,X2MouseButton',
          max_target_distance: 90
        },
        mouse: {
          move: 'makcu',
          dpi: 1100,
          sensitivity: 3.0,
          fov_width: 40,
          fov_height: 40
        }
      }
    };
  },
  mounted() {
    this.connectWebSocket();
    this.scanModels();
    this.$nextTick(() => {
      this.drawOffsetCanvas();
    });
    this.setupLogCapture();
    
    window.addEventListener('resize', this.onWindowResize);
  },
  beforeDestroy() {
    if (this.ws) {
      this.ws.close();
    }
    if (this.wsReconnectTimer) {
      clearTimeout(this.wsReconnectTimer);
    }
    window.removeEventListener('resize', this.onWindowResize);
  },
  methods: {
    connectWebSocket() {
      const wsUrl = 'ws://127.0.0.1:8765';
      
      try {
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
          this.wsConnected = true;
          this.sendCommand('get-config');
        };
        
        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this.handleWebSocketMessage(message);
          } catch (e) {
            console.error('解析 WebSocket 消息失败:', e);
          }
        };
        
        this.ws.onclose = () => {
          this.wsConnected = false;
          this.wsReconnectTimer = setTimeout(() => {
            this.connectWebSocket();
          }, 3000);
        };
        
        this.ws.onerror = (error) => {
          console.error('WebSocket 错误:', error);
        };
      } catch (e) {
        console.error('创建 WebSocket 失败:', e);
      }
    },
    
    sendCommand(command, data = null) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        const message = { command };
        if (data !== null) {
          message.data = data;
        }
        this.ws.send(JSON.stringify(message));
        return true;
      }
      return false;
    },
    
    handleWebSocketMessage(message) {
      const { type, data } = message;
      
      switch (type) {
        case 'config':
          this.handleConfigReceived(data);
          break;
        case 'fps':
          this.handleFpsData(data);
          break;
        case 'video-frame':
          this.handleVideoFrame(data);
          break;
        case 'log':
          this.handleLogMessage(data);
          break;
        case 'detections':
          this.handleDetections(data);
          break;
        case 'models':
          this.modelList = data;
          break;
        case 'config-saved':
          if (data && data.success) {
            this.$message.success('配置已保存');
          } else {
            this.$message.error('保存失败: ' + (data && data.error ? data.error : '未知错误'));
          }
          this.saving = false;
          break;
        default:
          break;
      }
    },
    
    handleConfigReceived(config) {
      if (config.ai) {
        this.config.ai = { ...this.config.ai, ...config.ai };
      }
      if (config.capture) {
        this.config.capture = { ...this.config.capture, ...config.capture };
      }
      if (config.aim) {
        this.config.aim = { ...this.config.aim, ...config.aim };
      }
      if (config.mouse) {
        this.config.mouse = { ...this.config.mouse, ...config.mouse };
      }
      
      if (config.aim && config.aim.hotkeys) {
        this.selectedHotkeys = config.aim.hotkeys.split(',').filter(h => h);
      }
      
      if (config.capture) {
        this.videoEnabled = config.capture.ai_debug || false;
      }
      
      this.$nextTick(() => {
        this.drawOffsetCanvas();
      });
    },
    
    handleFpsData(data) {
      if (data.capture !== undefined) {
        this.captureFps = data.capture;
      }
      if (data.predict !== undefined) {
        this.predictFps = data.predict;
      }
    },
    
    handleVideoFrame(data) {
      if (this.videoEnabled) {
        this.videoFrame = data;
      }
    },
    
    handleLogMessage(message) {
      this.appendLog(message);
    },
    
    handleDetections(detections) {
      if (Array.isArray(detections)) {
        this.detectionCount = detections.length;
        this.detections = detections;
        
        requestAnimationFrame(() => {
          this.drawDetections();
        });
      }
    },
    
    onVideoLoad() {
      requestAnimationFrame(() => {
        this.drawDetections();
      });
    },
    
    drawDetections() {
      const canvas = this.$refs.detectionCanvas;
      const img = this.$el.querySelector('.video-frame');
      if (!canvas || !img || !this.videoFrame) return;
      
      if (!img.complete || img.naturalWidth === 0) {
        setTimeout(() => this.drawDetections(), 100);
        return;
      }
      
      const ctx = canvas.getContext('2d');
      const container = canvas.parentElement;
      
      canvas.width = container.clientWidth;
      canvas.height = container.clientHeight;
      
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      if (!this.detections || this.detections.length === 0) return;
      
      const imgNaturalWidth = img.naturalWidth || this.config.capture.window_width;
      const imgNaturalHeight = img.naturalHeight || this.config.capture.window_height;
      const imgRatio = imgNaturalWidth / imgNaturalHeight;
      const containerRatio = canvas.width / canvas.height;
      
      let scaleX, scaleY, offsetX = 0, offsetY = 0;
      
      if (imgRatio > containerRatio) {
        scaleX = canvas.width / imgNaturalWidth;
        scaleY = scaleX;
        offsetY = (canvas.height - imgNaturalHeight * scaleY) / 2;
      } else {
        scaleY = canvas.height / imgNaturalHeight;
        scaleX = scaleY;
        offsetX = (canvas.width - imgNaturalWidth * scaleX) / 2;
      }
      
      this.detections.forEach((det, index) => {
        const box = det.box;
        if (!box || box.length < 4) return;
        
        const x1 = box[0] * scaleX + offsetX;
        const y1 = box[1] * scaleY + offsetY;
        const x2 = box[2] * scaleX + offsetX;
        const y2 = box[3] * scaleY + offsetY;
        const width = x2 - x1;
        const height = y2 - y1;
        
        const classId = det.class_id || 0;
        const colors = ['#409eff', '#67c23a', '#e6a23c', '#f56c6c', '#909399', '#606266'];
        const color = colors[classId % colors.length];
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, width, height);
        
        ctx.fillStyle = color + '20';
        ctx.fillRect(x1, y1, width, height);
        
        const label = `类别${classId} ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = '12px -apple-system, BlinkMacSystemFont, sans-serif';
        const textWidth = ctx.measureText(label).width;
        const textHeight = 16;
        
        ctx.fillStyle = color;
        ctx.fillRect(x1, y1 - textHeight - 2, textWidth + 8, textHeight + 4);
        
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x1 + 4, y1 - 4);
      });
    },
    
    saveConfig() {
      this.saving = true;
      this.config.aim.hotkeys = this.selectedHotkeys.join(',');
      this.config.capture.ai_debug = this.videoEnabled;
      
      const sent = this.sendCommand('save-config', this.config);
      
      if (!sent) {
        this.$message.error('WebSocket 未连接，保存失败');
        this.saving = false;
        return;
      }
      
      setTimeout(() => {
        if (this.saving) {
          this.$message.error('保存超时，请重试');
          this.saving = false;
        }
      }, 5000);
    },
    
    autoApply() {
      this.config.aim.hotkeys = this.selectedHotkeys.join(',');
      this.config.capture.ai_debug = this.videoEnabled;
      
      this.sendCommand('apply-config', this.config);
    },
    
    scanModels() {
      this.sendCommand('scan-models');
    },
    
    toggleAim() {
      this.config.aim.auto = !this.config.aim.auto;
      this.autoApply();
    },
    
    onHotkeysChanged(value) {
      this.selectedHotkeys = value;
      this.autoApply();
    },
    
    onVideoDebugChanged(value) {
      this.config.capture.ai_debug = value;
      this.autoApply();
      if (!value) {
        this.videoFrame = '';
        this.detectionCount = 0;
        this.detections = [];
        const canvas = this.$refs.detectionCanvas;
        if (canvas) {
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
      }
    },
    
    onWindowResize() {
      this.$nextTick(() => {
        this.drawDetections();
      });
    },
    
    onOffsetChanged() {
      this.drawOffsetCanvas();
      this.autoApply();
    },
    
    drawOffsetCanvas() {
      const canvas = this.$refs.offsetCanvas;
      if (!canvas) return;
      
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;

      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);

      ctx.strokeStyle = '#e4e7ed';
      ctx.lineWidth = 1;
      
      for (let i = 0; i <= 10; i++) {
        const x = (width / 10) * i;
        const y = (height / 10) * i;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      const centerX = width / 2;
      const centerY = height / 2;
      
      ctx.strokeStyle = '#c0c4cc';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(width, centerY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = '#909399';
      ctx.lineWidth = 2;
      
      const headY = centerY - 45;
      ctx.beginPath();
      ctx.ellipse(centerX, headY, 12, 14, 0, 0, Math.PI * 2);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(centerX, headY + 14);
      ctx.lineTo(centerX, headY + 20);
      ctx.stroke();
      
      const bodyTop = headY + 20;
      const bodyHeight = 38;
      ctx.beginPath();
      ctx.roundRect(centerX - 16, bodyTop, 32, bodyHeight, 4);
      ctx.stroke();
      
      const shoulderY = bodyTop + 5;
      ctx.beginPath();
      ctx.moveTo(centerX - 16, shoulderY);
      ctx.lineTo(centerX - 28, shoulderY + 22);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(centerX + 16, shoulderY);
      ctx.lineTo(centerX + 28, shoulderY + 22);
      ctx.stroke();
      
      const hipY = bodyTop + bodyHeight;
      ctx.beginPath();
      ctx.moveTo(centerX - 8, hipY);
      ctx.lineTo(centerX - 10, hipY + 35);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(centerX + 8, hipY);
      ctx.lineTo(centerX + 10, hipY + 35);
      ctx.stroke();

      const offsetX = this.config.aim.body_x_offset;
      const offsetY = this.config.aim.body_y_offset;
      const aimX = centerX + offsetX * 50;
      const aimY = centerY + offsetY * 55;

      ctx.fillStyle = '#f56c6c';
      ctx.beginPath();
      ctx.arc(aimX, aimY, 6, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = '#f56c6c';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(aimX - 12, aimY);
      ctx.lineTo(aimX + 12, aimY);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(aimX, aimY - 12);
      ctx.lineTo(aimX, aimY + 12);
      ctx.stroke();
    },
    
    onCanvasMouseDown(e) {
      this.isDragging = true;
      this.updateOffsetFromMouse(e);
    },
    
    onCanvasMouseMove(e) {
      if (this.isDragging) {
        this.updateOffsetFromMouse(e);
      }
    },
    
    onCanvasMouseUp() {
      this.isDragging = false;
    },
    
    updateOffsetFromMouse(e) {
      const canvas = this.$refs.offsetCanvas;
      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) * (canvas.width / rect.width);
      const y = (e.clientY - rect.top) * (canvas.height / rect.height);
      
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      
      const rawX = Math.max(-1, Math.min(1, (x - centerX) / 50));
      const rawY = Math.max(-1, Math.min(1, (y - centerY) / 55));
      
      this.config.aim.body_x_offset = Math.round(rawX * 100) / 100;
      this.config.aim.body_y_offset = Math.round(rawY * 100) / 100;
      
      this.drawOffsetCanvas();
      this.autoApply();
    },
    
    getLogClass(log) {
      if (log.includes('[ERROR]')) return 'error';
      if (log.includes('[WARN]')) return 'warn';
      if (log.includes('成功') || log.includes('已连接') || log.includes('已保存')) return 'success';
      return '';
    },
    
    getLogTime(log) {
      const match = log.match(/^\[(\d{1,2}:\d{2}:\d{2})\]/);
      return match ? match[1] : '';
    },
    
    getLogText(log) {
      return log.replace(/^\[\d{1,2}:\d{2}:\d{2}\]\s*/, '');
    },
    
    setupLogCapture() {
      const originalLog = console.log;
      const originalError = console.error;
      const originalWarn = console.warn;

      console.log = (...args) => {
        this.appendLog(args.join(' '));
        originalLog.apply(console, args);
      };

      console.error = (...args) => {
        this.appendLog('[ERROR] ' + args.join(' '));
        originalError.apply(console, args);
      };

      console.warn = (...args) => {
        this.appendLog('[WARN] ' + args.join(' '));
        originalWarn.apply(console, args);
      };
    },
    
    appendLog(message) {
      const timestamp = new Date().toLocaleTimeString();
      this.logs.push(`[${timestamp}] ${message}`);
      if (this.logs.length > 500) {
        this.logs.shift();
      }
      this.$nextTick(() => {
        const logContent = this.$refs.logContent;
        if (logContent) {
          logContent.scrollTop = logContent.scrollHeight;
        }
      });
    }
  }
});
