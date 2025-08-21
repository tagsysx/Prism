#!/bin/bash
# 使用screen会话运行Sionna训练，可以重新连接

# 检查screen是否安装
if ! command -v screen &> /dev/null; then
    echo "Screen not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y screen
fi

# 创建一个名为"prism_sionna_training"的screen会话
screen -dmS prism_sionna_training bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate py38
    cd /home/young/projects/prism/Prism
    python scripts/sionna_runner.py --mode train --config configs/ofdm-5g-sionna.yml --epochs 100 --batch_size 32 --device cuda --save_dir checkpoints/sionna_5g --results_dir results/sionna_5g
    echo 'Sionna training completed. Press any key to exit.'
    read
"

echo "Sionna training started in screen session 'prism_sionna_training'"
echo "To reconnect: screen -r prism_sionna_training"
echo "To detach from session: Ctrl+A then D"
echo "To list sessions: screen -ls"
echo ""
echo "Training parameters:"
echo "  - Mode: train"
echo "  - Config: configs/ofdm-5g-sionna.yml"
echo "  - Epochs: 100"
echo "  - Batch size: 32"
echo "  - Device: cuda"
echo "  - Save dir: checkpoints/sionna_5g"
echo "  - Results dir: results/sionna_5g"
echo "  - Checkpoint frequency: Every 10 epochs"
echo ""
echo "Data split: 80% train, 0% validation, 20% test"
