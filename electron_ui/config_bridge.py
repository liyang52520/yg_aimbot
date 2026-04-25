#!/usr/bin/env python3
"""
配置桥接脚本 - 供 Electron 调用以读写配置
"""
import sys
import json
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.services.config_service import config_service


def get_config():
    """获取完整配置"""
    return {
        'ai': config_service.get_section('ai'),
        'capture': config_service.get_section('capture'),
        'aim': config_service.get_section('aim'),
        'mouse': config_service.get_section('mouse')
    }


def save_config(config):
    """保存配置"""
    config_service.update_section('ai', config.get('ai', {}))
    config_service.update_section('capture', config.get('capture', {}))
    config_service.update_section('aim', config.get('aim', {}))
    config_service.update_section('mouse', config.get('mouse', {}))
    config_service.save()
    return {'success': True}


def apply_config(config):
    """应用配置到内存"""
    config_service.update_section('ai', config.get('ai', {}))
    config_service.update_section('capture', config.get('capture', {}))
    config_service.update_section('aim', config.get('aim', {}))
    config_service.update_section('mouse', config.get('mouse', {}))
    return {'success': True}


def scan_models():
    """扫描模型文件"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    extensions = ['.pt', '.onnx', '.engine']
    
    models = []
    try:
        if os.path.exists(models_dir) and os.path.isdir(models_dir):
            for file in os.listdir(models_dir):
                if any(file.endswith(ext) for ext in extensions):
                    models.append(file)
    except Exception as e:
        print(f"扫描模型失败: {e}", file=sys.stderr)
    
    if not models:
        models = ['YOLOv8s_apex_teammate_enemy.engine']
    
    return models


def main():
    """主函数 - 从 stdin 读取命令，输出到 stdout"""
    try:
        # 读取输入
        line = sys.stdin.readline()
        if not line:
            return
        
        request = json.loads(line.strip())
        command = request.get('command')
        data = request.get('data', {})
        
        # 执行命令
        if command == 'get-config':
            result = get_config()
        elif command == 'save-config':
            result = save_config(data)
        elif command == 'apply-config':
            result = apply_config(data)
        elif command == 'scan-models':
            result = scan_models()
        else:
            result = {'error': f'Unknown command: {command}'}
        
        # 输出结果
        print(json.dumps(result))
        sys.stdout.flush()
        
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.stdout.flush()


if __name__ == '__main__':
    main()
