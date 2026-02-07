import typing
import torch
import torch.distributed as dist
import os

# 信任 vbench 的 checkpoint 的前提下，加入安全白名单
torch.serialization.add_safe_globals([typing.OrderedDict])

# ===== 添加分布式初始化 =====
def init_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',  # GPU 通信用 nccl
            init_method='env://',
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f'cuda:{local_rank}')
        )
        
        # 设置当前进程使用的 GPU
        torch.cuda.set_device(local_rank)
        
        print(f"[Rank {rank}/{world_size}] 初始化完成，使用 GPU {local_rank}")
        return local_rank
    else:
        print("未检测到分布式环境，使用单 GPU 模式")
        return 0

local_rank = init_distributed()
device = f"cuda:{local_rank}"


from vbench2_beta_i2v import VBenchI2V

my_VBench = VBenchI2V("cuda", "VBench/vbench2_beta_i2v/vbench2_i2v_full_info.json", "save_results/vbench_wan2.2_i2v_14b_4steps_flashattn2_woquant/evaluation_results")
my_VBench.evaluate(
    videos_path = "save_results/vbench_wan2.2_i2v_14b_4steps_flashattn2_woquant",
    name = "vbench_wan2.2_i2v_14b_4steps_flashattn2_woquant",
    dimension_list = ["dynamic_degree", "imaging_quality"],
    resolution = "16-9"
)


# ===== 清理分布式环境 =====
if dist.is_initialized():
    dist.destroy_process_group()