#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频图像质量评估脚本
使用 MUSIQ 模型对文件夹中的 mp4 视频进行图像质量评估
"""

import os
import json
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
from torchvision import transforms


def load_video(video_path, max_frames=None):
    """
    加载视频并返回帧张量
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大采样帧数，None表示加载所有帧
        
    Returns:
        torch.Tensor: 形状为 (T, C, H, W) 的张量
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 确定要采样的帧索引
    if max_frames is not None and max_frames < total_frames:
        # 均匀采样
        frame_indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        frame_indices = np.arange(total_frames)
    
    frames = []
    current_idx = 0
    
    for target_idx in frame_indices:
        # 跳到目标帧
        if target_idx != current_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        
        ret, frame = cap.read()
        if not ret:
            break
            
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        current_idx = target_idx + 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"无法从视频中读取任何帧: {video_path}")
    
    # 转换为张量 (T, C, H, W)
    frames = np.stack(frames, axis=0)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
    
    return frames


def transform_images(images, preprocess_mode='longer'):
    """
    预处理图像
    
    Args:
        images: 形状为 (T, C, H, W) 的张量
        preprocess_mode: 预处理模式
            - 'shorter': 将短边缩放到512
            - 'shorter_centercrop': 将短边缩放到512后中心裁剪
            - 'longer': 将长边缩放到512
            - 'None': 不缩放
            
    Returns:
        归一化后的图像张量
    """
    if preprocess_mode.startswith('shorter'):
        _, _, h, w = images.size()
        if min(h, w) > 512:
            scale = 512. / min(h, w)
            new_size = (int(scale * h), int(scale * w))
            images = transforms.Resize(size=new_size, antialias=True)(images)
            if preprocess_mode == 'shorter_centercrop':
                images = transforms.CenterCrop(512)(images)

    elif preprocess_mode == 'longer':
        _, _, h, w = images.size()
        if max(h, w) > 512:
            scale = 512. / max(h, w)
            new_size = (int(scale * h), int(scale * w))
            images = transforms.Resize(size=new_size, antialias=True)(images)

    elif preprocess_mode == 'None':
        return images / 255.

    else:
        raise ValueError(f"不支持的预处理模式: {preprocess_mode}")
    
    return images / 255.


def get_video_list(folder_path):
    """获取文件夹中所有mp4视频文件"""
    video_extensions = ('.mp4', '.MP4')
    video_list = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(video_extensions):
            video_path = os.path.join(folder_path, filename)
            video_list.append(video_path)
    
    video_list.sort()
    return video_list


def evaluate_technical_quality(model, video_list, device, preprocess_mode='longer', 
                                max_frames=None):
    """
    评估视频列表的技术质量
    
    Args:
        model: MUSIQ 模型
        video_list: 视频路径列表
        device: 计算设备
        preprocess_mode: 预处理模式
        max_frames: 每个视频最大采样帧数
        
    Returns:
        average_score: 平均分数 (归一化到0-1)
        video_results: 每个视频的详细结果
    """
    video_results = []
    
    for video_path in tqdm(video_list, desc="评估视频"):
        try:
            # 获取文件名（作为prompt）
            filename = os.path.basename(video_path)
            prompt = os.path.splitext(filename)[0]
            
            # 加载视频
            images = load_video(video_path, max_frames=max_frames)
            
            # 预处理
            images = transform_images(images, preprocess_mode)
            
            # 对每一帧评分
            frame_scores = []
            with torch.no_grad():
                for i in range(len(images)):
                    frame = images[i].unsqueeze(0).to(device)
                    score = model(frame)
                    frame_scores.append(float(score))
            
            avg_score = sum(frame_scores) / len(frame_scores)
            
            result = {
                'video_path': video_path,
                'prompt': prompt,
                'video_results': avg_score,
                'num_frames': len(frame_scores),
                'min_score': min(frame_scores),
                'max_score': max(frame_scores),
            }
            video_results.append(result)
            
        except Exception as e:
            print(f"\n警告: 处理视频 {video_path} 时出错: {e}")
            continue
    
    if len(video_results) == 0:
        return 0.0, []
    
    # 计算平均分数并归一化到0-1
    average_score = sum([r['video_results'] for r in video_results]) / len(video_results)
    average_score_normalized = average_score / 100.0
    
    return average_score_normalized, video_results


def compute_imaging_quality(video_folder, device='cuda', preprocess_mode='longer',
                            max_frames=None, model_path=None):
    """
    计算视频文件夹的图像质量
    
    Args:
        video_folder: 视频文件夹路径
        device: 计算设备 ('cuda' 或 'cpu')
        preprocess_mode: 预处理模式
        max_frames: 每个视频最大采样帧数
        model_path: 模型权重路径（可选，不提供则自动下载）
        
    Returns:
        all_results: 归一化的平均分数 (0-1)
        video_results: 每个视频的详细结果
    """
    # 使用 pyiqa 加载模型
    try:
        import pyiqa
        print("使用 pyiqa 加载 MUSIQ 模型...")
        model = pyiqa.create_metric('musiq', device=device)
    except ImportError:
        # 如果 pyiqa 不可用，尝试直接使用 MUSIQ
        print("pyiqa 不可用，尝试直接加载 MUSIQ 模型...")
        try:
            from pyiqa.archs.musiq_arch import MUSIQ
            model = MUSIQ(pretrained_model_path=model_path)
            model.to(device)
            model.eval()
        except Exception as e:
            raise RuntimeError(
                f"无法加载模型。请安装 pyiqa: pip install pyiqa\n错误: {e}"
            )
    
    # 获取视频列表
    video_list = get_video_list(video_folder)
    
    if len(video_list) == 0:
        raise ValueError(f"在 {video_folder} 中未找到任何 mp4 文件")
    
    print(f"找到 {len(video_list)} 个视频文件")
    
    # 评估
    all_results, video_results = evaluate_technical_quality(
        model=model,
        video_list=video_list,
        device=device,
        preprocess_mode=preprocess_mode,
        max_frames=max_frames
    )
    
    return all_results, video_results


def print_results(all_results, video_results):
    """打印评估结果"""
    print("\n" + "=" * 70)
    print("图像质量评估结果")
    print("=" * 70)
    
    # 按分数排序
    sorted_results = sorted(video_results, key=lambda x: x['video_results'], reverse=True)
    
    for i, r in enumerate(sorted_results, 1):
        print(f"\n[{i}] 视频: {os.path.basename(r['video_path'])}")
        print(f"    Prompt: {r['prompt'][:50]}..." if len(r['prompt']) > 50 else f"    Prompt: {r['prompt']}")
        print(f"    平均分数: {r['video_results']:.4f}")
        print(f"    分数范围: [{r['min_score']:.2f}, {r['max_score']:.2f}]")
        print(f"    评估帧数: {r['num_frames']}")
    
    print("\n" + "=" * 70)
    print(f"总体平均分数 (原始): {all_results * 100:.4f}")
    print(f"总体平均分数 (归一化 0-1): {all_results:.4f}")
    print(f"评估视频数量: {len(video_results)}")
    print("=" * 70)


def save_results(all_results, video_results, output_path):
    """保存评估结果到JSON文件"""
    output_data = {
        'overall_score': all_results,
        'overall_score_raw': all_results * 100,
        'num_videos': len(video_results),
        'video_results': video_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='视频图像质量评估工具 (基于MUSIQ模型)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python imaging_quality.py --video_folder ./videos
  python imaging_quality.py --video_folder ./videos --max_frames 16 --device cpu
  python imaging_quality.py --video_folder ./videos --output results.json
        """
    )
    
    parser.add_argument(
        '--video_folder', 
        type=str, 
        required=True,
        help='包含mp4视频的文件夹路径'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        choices=['cuda', 'cpu'],
        help='计算设备 (默认: cuda)'
    )
    
    parser.add_argument(
        '--preprocess_mode', 
        type=str, 
        default='longer',
        choices=['shorter', 'shorter_centercrop', 'longer', 'None'],
        help='图像预处理模式 (默认: longer)'
    )
    
    parser.add_argument(
        '--max_frames', 
        type=int, 
        default=None,
        help='每个视频最大采样帧数，None表示使用所有帧 (默认: None)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='imaging_quality_results.json',
        help='输出JSON文件路径 (默认: imaging_quality_results.json)'
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=None,
        help='MUSIQ模型权重路径（可选，不提供则自动下载）'
    )
    
    args = parser.parse_args()
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA 不可用，切换到 CPU")
        args.device = 'cpu'
    
    # 检查文件夹是否存在
    if not os.path.isdir(args.video_folder):
        raise ValueError(f"文件夹不存在: {args.video_folder}")
    
    print(f"设备: {args.device}")
    print(f"预处理模式: {args.preprocess_mode}")
    print(f"最大采样帧数: {args.max_frames if args.max_frames else '全部'}")
    print()
    
    # 运行评估
    all_results, video_results = compute_imaging_quality(
        video_folder=args.video_folder,
        device=args.device,
        preprocess_mode=args.preprocess_mode,
        max_frames=args.max_frames,
        model_path=args.model_path
    )
    
    # 打印结果
    print_results(all_results, video_results)
    
    # 保存结果
    save_results(all_results, video_results, args.output)


if __name__ == "__main__":
    main()