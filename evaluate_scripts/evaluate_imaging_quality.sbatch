#!/bin/bash
#SBATCH -J evaluate_imaging_quality  # 作业名称
#SBATCH -N 1                         # 节点数
#SBATCH -n 1                         # 总任务数
#SBATCH --ntasks-per-node=1          # 每节点任务数
#SBATCH --cpus-per-task=10           # 每个任务使用的 CPU 数
#SBATCH --gres=gpu:1                 # GPU 数量
#SBATCH --mem=60G                    # 内存
#SBATCH -t 7-0:0:0                   # 最大运行时间 (7天)
#SBATCH -p NV_RTX4090                   # 队列/分区
#SBATCH --output=logs/%x-%j.out      # 标准输出日志
#SBATCH --error=logs/%x-%j.err       # 标准错误日志

# ======== 作业正式开始 ========

set -eo pipefail
# set -x

# 确保日志目录存在（若集群允许，可在这里创建；也可以提前创建）
mkdir -p logs

# 模块及环境设置 (根据你的实际情况修改)
module purge


# Conda 环境激活 (若使用 conda/mamba)
source /public/home/cit_yunkangtao/anaconda3/etc/profile.d/conda.sh
conda activate vbench

python evaluate_scripts/evaluate_imaging_quality.py \
    --video_folder save_results/i2v_mergedmd_14b_480P \
    --output save_results/i2v_mergedmd_14b_480P/imaging_quality.json

# ======== 作业结束 ========