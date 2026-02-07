#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文本-视频对齐评估脚本
使用ViCLIP模型评估视频与其对应prompt的对齐程度
视频文件名即为prompt（不含.mp4后缀）
"""

import os
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

from vbench.utils import clip_transform, read_frames_decord_by_fps, CACHE_DIR
from vbench.third_party.ViCLIP.viclip import ViCLIP
from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer


def get_text_features(model, input_text, tokenizer, text_feature_dict={}):
    """获取文本特征向量"""
    if input_text in text_feature_dict:
        return text_feature_dict[input_text]
    text_template = f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_feature_dict[input_text] = text_features
    return text_features


def get_vid_features(model, input_frames):
    """获取视频特征向量"""
    with torch.no_grad():
        clip_feat = model.encode_vision(input_frames, test=True).float()
        clip_feat /= clip_feat.norm(dim=-1, keepdim=True)
    return clip_feat


def extract_prompt_from_filename(filename):
    """
    从文件名提取prompt
    例如: "a cat running in the garden.mp4" -> "a cat running in the garden"
    """
    prompt = os.path.splitext(filename)[0]
    # 如果文件名中有下划线，可以选择替换为空格（根据你的命名规则调整）
    # prompt = prompt.replace('_', ' ')
    return prompt


def load_viclip_model(device):
    """加载ViCLIP模型"""
    print("=" * 50)
    print("Loading ViCLIP model...")
    print("=" * 50)
    
    # Tokenizer路径
    tokenizer_path = os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz")
    
    # 模型权重路径
    pretrain_path = os.path.join(CACHE_DIR, "ViCLIP/ViClip-InternVid-10M-FLT.pth")
    
    # 检查文件是否存在
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}\n"
            "Please download ViCLIP model files first.\n"
            "You can use: python -c \"from vbench import VBench; VBench.download_all_resources()\""
        )
    
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(
            f"Model weights not found at {pretrain_path}\n"
            "Please download ViCLIP model files first."
        )
    
    tokenizer = SimpleTokenizer(tokenizer_path)
    
    viclip = ViCLIP(
        tokenizer=tokenizer,
        pretrain=pretrain_path
    ).to(device)
    viclip.eval()
    
    print("ViCLIP model loaded successfully!")
    return viclip, tokenizer


def evaluate_text_video_alignment(
    video_folder, 
    device, 
    num_frames=8,
    sample="middle"
):
    """
    评估文件夹中所有视频的文本-视频对齐程度
    
    Args:
        video_folder: 包含MP4文件的文件夹路径
        device: 计算设备
        num_frames: 采样帧数
        sample: 采样策略 ('middle', 'uniform')
    
    Returns:
        avg_score: 平均对齐分数
        video_results: 每个视频的详细结果
    """
    
    # 加载模型
    viclip, tokenizer = load_viclip_model(device)
    
    # 图像变换
    image_transform = clip_transform(224)
    
    # 获取所有MP4文件
    video_files = sorted([
        f for f in os.listdir(video_folder) 
        if f.lower().endswith('.mp4')
    ])
    
    if len(video_files) == 0:
        print(f"Error: No MP4 files found in {video_folder}")
        return None, []
    
    print(f"\nFound {len(video_files)} video files to evaluate")
    print("-" * 50)
    
    sim_scores = []
    video_results = []
    text_feature_dict = {}  # 缓存文本特征
    failed_videos = []
    
    for video_file in tqdm(video_files, desc="Evaluating videos"):
        video_path = os.path.join(video_folder, video_file)
        prompt = extract_prompt_from_filename(video_file)
        
        try:
            with torch.no_grad():
                # 读取视频帧
                images = read_frames_decord_by_fps(
                    video_path, 
                    num_frames=num_frames, 
                    sample=sample
                )
                images = image_transform(images)
                images = images.to(device)
                
                # 获取视频特征
                clip_feat = get_vid_features(viclip, images.unsqueeze(0))
                
                # 获取文本特征
                text_feat = get_text_features(
                    viclip, prompt, tokenizer, text_feature_dict
                )
                
                # 计算余弦相似度
                logit_per_text = clip_feat @ text_feat.T
                score = float(logit_per_text[0][0].cpu())
                
                sim_scores.append(score)
                video_results.append({
                    'video_path': video_path,
                    'video_name': video_file,
                    'prompt': prompt,
                    'alignment_score': score
                })
                
        except Exception as e:
            print(f"\nError processing {video_file}: {e}")
            failed_videos.append({
                'video_name': video_file,
                'error': str(e)
            })
            continue
    
    # 计算统计信息
    if sim_scores:
        avg_score = float(np.mean(sim_scores))
        std_score = float(np.std(sim_scores))
        min_score = float(np.min(sim_scores))
        max_score = float(np.max(sim_scores))
    else:
        avg_score = std_score = min_score = max_score = 0
    
    stats = {
        'average': avg_score,
        'std': std_score,
        'min': min_score,
        'max': max_score,
        'num_evaluated': len(sim_scores),
        'num_failed': len(failed_videos)
    }
    
    return stats, video_results, failed_videos


def print_results(stats, video_results):
    """打印评估结果"""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n📊 Statistics:")
    print(f"   • Videos evaluated: {stats['num_evaluated']}")
    print(f"   • Videos failed:    {stats['num_failed']}")
    print(f"   • Average score:    {stats['average']:.4f}")
    print(f"   • Std deviation:    {stats['std']:.4f}")
    print(f"   • Min score:        {stats['min']:.4f}")
    print(f"   • Max score:        {stats['max']:.4f}")
    
    if video_results:
        # 按分数排序
        sorted_results = sorted(
            video_results, 
            key=lambda x: x['alignment_score'], 
            reverse=True
        )
        
        # 显示最高分
        print(f"\n🏆 Top 5 Best Aligned Videos:")
        for i, res in enumerate(sorted_results[:5], 1):
            prompt_display = res['prompt'][:60] + "..." if len(res['prompt']) > 60 else res['prompt']
            print(f"   {i}. [{res['alignment_score']:.4f}] {prompt_display}")
        
        # 显示最低分
        print(f"\n⚠️  Top 5 Worst Aligned Videos:")
        for i, res in enumerate(sorted_results[-5:], 1):
            prompt_display = res['prompt'][:60] + "..." if len(res['prompt']) > 60 else res['prompt']
            print(f"   {i}. [{res['alignment_score']:.4f}] {prompt_display}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate text-video alignment using ViCLIP (VBench)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_alignment.py --video_folder ./generated_videos
  python evaluate_alignment.py --video_folder ./videos --output results.json --device cuda:0
  python evaluate_alignment.py --video_folder ./videos --num_frames 16 --sample uniform
        """
    )
    
    parser.add_argument(
        '--video_folder', 
        type=str, 
        required=True,
        help='Path to folder containing MP4 files (filename = prompt)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='alignment_results.json',
        help='Output JSON file path (default: alignment_results.json)'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='cuda',
        help='Device to use: cuda, cuda:0, cuda:1, or cpu (default: cuda)'
    )
    parser.add_argument(
        '--num_frames', 
        type=int, 
        default=8,
        help='Number of frames to sample from each video (default: 8)'
    )
    parser.add_argument(
        '--sample', 
        type=str, 
        default='middle',
        choices=['middle', 'uniform'],
        help='Frame sampling strategy (default: middle)'
    )
    
    args = parser.parse_args()
    
    # 检查视频文件夹
    if not os.path.isdir(args.video_folder):
        print(f"Error: Video folder not found: {args.video_folder}")
        return
    
    # 检查设备
    if 'cuda' in args.device and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU instead")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # 执行评估
    stats, video_results, failed_videos = evaluate_text_video_alignment(
        video_folder=args.video_folder,
        device=device,
        num_frames=args.num_frames,
        sample=args.sample
    )
    
    if stats is None:
        return
    
    # 打印结果
    print_results(stats, video_results)
    
    # 保存结果到JSON
    output_data = {
        'config': {
            'video_folder': os.path.abspath(args.video_folder),
            'num_frames': args.num_frames,
            'sample_strategy': args.sample,
            'device': str(device)
        },
        'statistics': stats,
        'video_results': video_results,
        'failed_videos': failed_videos
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"📁 Results saved to: {args.output}")


if __name__ == '__main__':
    main()