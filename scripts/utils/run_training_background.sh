#!/bin/bash
# 使用screen会话运行Prism训练，可以重新连接

# 默认参数
CONFIG_FILE="configs/ofdm-5g-sionna.yml"
EPOCHS=100
BATCH_SIZE=32
DEVICE="cuda"
SESSION_NAME="prism_training"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --session)
            SESSION_NAME="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --config FILE       Configuration file (default: configs/ofdm-5g-sionna.yml)"
            echo "  --epochs N          Number of epochs (default: 100)"
            echo "  --batch-size N      Batch size (default: 32)"
            echo "  --device DEVICE     Device to use (default: cuda)"
            echo "  --session NAME      Screen session name (default: prism_training)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 检查screen是否安装
if ! command -v screen &> /dev/null; then
    echo "Screen not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y screen
fi

# 检查会话是否已存在
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Screen session '$SESSION_NAME' already exists!"
    echo "To reconnect: screen -r $SESSION_NAME"
    echo "To kill existing session: screen -S $SESSION_NAME -X quit"
    exit 1
fi

# 创建screen会话
screen -dmS "$SESSION_NAME" bash -c "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate py38
    cd /home/young/projects/prism/Prism
    python scripts/prism_runner.py --mode train --config $CONFIG_FILE --epochs $EPOCHS --batch_size $BATCH_SIZE --device $DEVICE --save_dir checkpoints/sionna_5g --results_dir results/sionna_5g
    echo 'Prism training completed. Press any key to exit.'
    read
"

echo "Prism training started in screen session '$SESSION_NAME'"
echo "To reconnect: screen -r $SESSION_NAME"
echo "To detach from session: Ctrl+A then D"
echo "To list sessions: screen -ls"
echo ""
echo "Training parameters:"
echo "  - Mode: train"
echo "  - Config: $CONFIG_FILE"
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Device: $DEVICE"
echo "  - Save dir: checkpoints/sionna_5g"
echo "  - Results dir: results/sionna_5g"
echo "  - Session name: $SESSION_NAME"
echo ""
echo "Monitoring commands:"
echo "  - Monitor training: python scripts/utils/monitor_training.py"
echo "  - Read checkpoint: python scripts/utils/read_checkpoint.py <checkpoint_file>"
echo ""
echo "Data split: 80% train, 0% validation, 20% test"
