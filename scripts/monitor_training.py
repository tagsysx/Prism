#!/usr/bin/env python3
"""
训练监控脚本 - 监控训练进度和系统资源使用情况
"""

import os
import time
import psutil
import json
from pathlib import Path
import argparse
import torch

def get_training_process():
    """查找训练进程"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'] and 'sionna_runner.py' in ' '.join(proc.info['cmdline']):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None

def monitor_training(interval=30):
    """监控训练进程"""
    print("=== Prism Training Monitor ===")
    print(f"Monitoring interval: {interval} seconds")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # 检查训练进程
            proc = get_training_process()
            
            if proc:
                try:
                    # 获取进程信息
                    cpu_percent = proc.cpu_percent()
                    memory_info = proc.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
                    print(f"  Training Process: PID {proc.pid}")
                    print(f"  CPU Usage: {cpu_percent:.1f}%")
                    print(f"  Memory Usage: {memory_mb:.1f} MB")
                    print(f"  Status: {proc.status()}")
                    
                    # 检查GPU使用情况（如果nvidia-smi可用）
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                                               '--format=csv,noheader,nounits'], 
                                              capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            gpu_info = result.stdout.strip().split(', ')
                            print(f"  GPU Usage: {gpu_info[0]}%")
                            print(f"  GPU Memory: {gpu_info[1]}/{gpu_info[2]} MB")
                    except:
                        pass
                    
                    # 检查训练进度
                    checkpoint_dir = Path('checkpoints/sionna_5g')
                    if checkpoint_dir.exists():
                        # 检查最佳模型文件
                        best_model = checkpoint_dir / 'best_model.pth'
                        if best_model.exists():
                            try:
                                checkpoint = torch.load(best_model, map_location='cpu')
                                epoch = checkpoint.get('epoch', 'N/A')
                                loss = checkpoint.get('loss', 'N/A')
                                print(f"  Training Progress: Epoch {epoch}")
                                if isinstance(loss, (int, float)):
                                    print(f"  Best Loss: {loss:.6f}")
                                else:
                                    print(f"  Best Loss: {loss}")
                            except:
                                print(f"  Training Progress: Could not read checkpoint")
                        
                        # 检查epoch检查点文件
                        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
                        if checkpoints:
                            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                            epoch = latest_checkpoint.stem.split('_')[-1]
                            print(f"  Latest Epoch Checkpoint: {epoch}")
                    
                    # 检查结果文件
                    results_file = Path('results/sionna_5g/test_results.pt')
                    if results_file.exists():
                        try:
                            results = torch.load(results_file)
                            print(f"  Test Results: MSE={results.get('mse', 'N/A'):.4f}, "
                                  f"MAE={results.get('mae', 'N/A'):.4f}")
                        except:
                            print("  Test Results: Could not load results file")
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Process no longer accessible")
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] No training process found")
                
                # 检查是否训练已完成
                results_file = Path('results/sionna_5g/test_results.pt')
                if results_file.exists():
                    print("  Training appears to be completed!")
                    try:
                        results = torch.load(results_file)
                        print(f"  Final Results: MSE={results['mse']:.4f}, "
                              f"MAE={results['mae']:.4f}, RMSE={results['rmse']:.4f}")
                        break
                    except:
                        print("  Could not load final results")
                        break
            
            print("-" * 50)
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")

def main():
    parser = argparse.ArgumentParser(description='Monitor Prism training progress')
    parser.add_argument('--interval', '-i', type=int, default=30, 
                       help='Monitoring interval in seconds (default: 30)')
    
    args = parser.parse_args()
    monitor_training(args.interval)

if __name__ == '__main__':
    main()
