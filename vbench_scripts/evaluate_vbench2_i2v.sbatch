#!/bin/bash
#SBATCH -J evaluate_videos_i2v           # 作业名称
#SBATCH -N 1                         # 节点数
#SBATCH -n 1                         # 总任务数
#SBATCH --ntasks-per-node=1          # 每节点任务数
#SBATCH --cpus-per-task=20           # 每个任务使用的 CPU 数
#SBATCH --gres=gpu:1                 # GPU 数量
#SBATCH --mem=100G                    # 内存
#SBATCH -t 7-0:0:0                   # 最大运行时间 (7天)
#SBATCH -p NV_RTX4090                   # 队列/分区
#SBATCH --output=logs/%x-%j.out      # 标准输出日志
#SBATCH --error=logs/%x-%j.err       # 标准错误日志

# ======== 作业正式开始 ========

set -euo pipefail
# set -x

# 确保日志目录存在（若集群允许，可在这里创建；也可以提前创建）
# mkdir -p logs

# 模块及环境设置 (根据你的实际情况修改)
module purge
# module load cuda/12.1    # 示例：若需加载 CUDA 模块请取消注释并修改版本
# module load gcc/11.2     # 示例：如需特定编译器

# Conda 环境激活 (若使用 conda/mamba)
# source ~/.bashrc           # 确保加载 shell 初始化脚本
source /public/home/cit_yunkangtao/anaconda3/etc/profile.d/conda.sh
conda activate vbench      # 修改为你的环境名称

# 切换到项目目录 (如果提交时不在项目根目录，记得修改)
# cd /path/to/your/project

# 设置GPU数量
NGPUS=1  # 根据你申请的GPU数量修改

export OMP_NUM_THREADS=4
export TORCH_HUB_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 执行推理脚本
# bash evaluate_vbench.sh
torchrun --nproc_per_node=${NGPUS} --standalone vbench_scripts/evaluate_vbench2_i2v.py

# ======== 作业结束 ========