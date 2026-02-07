#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频动态程度评估脚本
使用 RAFT 光流模型评估视频的动态程度
"""

import argparse
import os
import sys
import cv2
import glob
import numpy as np
import torch
from tqdm import tqdm
from easydict import EasyDict as edict
import json
from datetime import datetime


# ============== RAFT 模型相关 ==============
# 需要先克隆 RAFT 仓库并添加到路径
# git clone https://github.com/princeton-vl/RAFT.git

def setup_raft_path(raft_path=None):
    """设置 RAFT 路径"""
    if raft_path is None:
        # 默认在当前目录或上级目录查找
        possible_paths = [
            './RAFT',
            '../RAFT',
            './RAFT/core',
            os.path.expanduser('~/RAFT'),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                raft_path = p
                break
    
    if raft_path is None:
        raise RuntimeError(
            "找不到 RAFT 目录。请先克隆 RAFT 仓库:\n"
            "git clone https://github.com/princeton-vl/RAFT.git"
        )
    
    # 添加到系统路径
    core_path = os.path.join(raft_path, 'core') if 'core' not in raft_path else raft_path
    if core_path not in sys.path:
        sys.path.insert(0, raft_path)
        sys.path.insert(0, core_path)
    
    return raft_path


class DynamicDegree:
    """视频动态程度评估器（使用RAFT模型）"""
    
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.params = None
        self.load_model()
    
    def load_model(self):
        """加载 RAFT 模型"""
        from vbench.third_party.RAFT.core.raft import RAFT
        from vbench.third_party.RAFT.core.utils_core.utils import InputPadder
        self.InputPadder = InputPadder
        
        self.model = RAFT(self.args)
        
        if not os.path.exists(self.args.model):
            raise FileNotFoundError(
                f"找不到模型文件: {self.args.model}\n"
                "请下载 RAFT 预训练模型:\n"
                "wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip\n"
                "unzip models.zip"
            )
        
        ckpt = torch.load(self.args.model, map_location="cpu")
        # 处理 DataParallel 保存的模型
        new_ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
        self.model.load_state_dict(new_ckpt)
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ RAFT 模型加载成功: {self.args.model}")
    
    def get_score(self, img, flo):
        """计算光流分数（取 top 5% 光流幅度均值）"""
        img = img[0].permute(1, 2, 0).cpu().numpy()
        flo = flo[0].permute(1, 2, 0).cpu().numpy()
        
        u = flo[:, :, 0]
        v = flo[:, :, 1]
        rad = np.sqrt(np.square(u) + np.square(v))
        
        h, w = rad.shape
        rad_flat = rad.flatten()
        cut_index = int(h * w * 0.05)
        
        max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])
        
        return max_rad.item()
    
    def set_params(self, frame, count):
        """根据视频分辨率和帧数设置动态阈值参数"""
        scale = min(list(frame.shape)[-2:])
        self.params = {
            "thres": 6.0 * (scale / 256.0),
            "count_num": round(4 * (count / 16.0))
        }
    
    def infer(self, video_path):
        """
        推理单个视频
        
        Returns:
            dict: 包含动态程度评估结果
        """
        with torch.no_grad():
            if video_path.endswith('.mp4') or video_path.endswith('.avi') or video_path.endswith('.mov'):
                frames = self.get_frames(video_path)
            elif os.path.isdir(video_path):
                frames = self.get_frames_from_img_folder(video_path)
            else:
                raise NotImplementedError(f"不支持的格式: {video_path}")
            
            if len(frames) < 2:
                return {
                    'is_dynamic': False,
                    'flow_scores': [],
                    'mean_flow_score': 0.0,
                    'max_flow_score': 0.0,
                    'num_frames': len(frames)
                }
            
            self.set_params(frame=frames[0], count=len(frames))
            
            flow_scores = []
            for image1, image2 in zip(frames[:-1], frames[1:]):
                padder = self.InputPadder(image1.shape)
                image1, image2 = padder.pad(image1, image2)
                _, flow_up = self.model(image1, image2, iters=20, test_mode=True)
                max_rad = self.get_score(image1, flow_up)
                flow_scores.append(max_rad)
            
            is_dynamic = self.check_move(flow_scores)
            
            return {
                'is_dynamic': is_dynamic,
                'flow_scores': flow_scores,
                'mean_flow_score': float(np.mean(flow_scores)) if flow_scores else 0.0,
                'max_flow_score': float(np.max(flow_scores)) if flow_scores else 0.0,
                'min_flow_score': float(np.min(flow_scores)) if flow_scores else 0.0,
                'std_flow_score': float(np.std(flow_scores)) if flow_scores else 0.0,
                'threshold': self.params['thres'],
                'count_threshold': self.params['count_num'],
                'num_frames': len(frames)
            }
    
    def check_move(self, score_list):
        """判断视频是否为动态"""
        thres = self.params["thres"]
        count_num = self.params["count_num"]
        count = 0
        for score in score_list:
            if score > thres:
                count += 1
            if count >= count_num:
                return True
        return False
    
    def get_frames(self, video_path):
        """从视频文件中提取帧"""
        frame_list = []
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
        
        fps = video.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        
        interval = max(1, round(fps / 8))  # 采样到约 8fps
        
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
                frame = frame[None].to(self.device)
                frame_list.append(frame)
            else:
                break
        video.release()
        
        if not frame_list:
            raise ValueError(f"视频没有有效帧: {video_path}")
        
        frame_list = self.extract_frame(frame_list, interval)
        return frame_list
    
    def extract_frame(self, frame_list, interval=1):
        """按间隔提取帧"""
        extract = []
        for i in range(0, len(frame_list), interval):
            extract.append(frame_list[i])
        return extract
    
    def get_frames_from_img_folder(self, img_folder):
        """从图片文件夹中加载帧"""
        exts = ['jpg', 'png', 'jpeg', 'bmp', 'tif', 'tiff', 
                'JPG', 'PNG', 'JPEG', 'BMP', 'TIF', 'TIFF']
        frame_list = []
        imgs = sorted([
            p for p in glob.glob(os.path.join(img_folder, "*")) 
            if os.path.splitext(p)[1][1:] in exts
        ])
        
        for img in imgs:
            frame = cv2.imread(img, cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame.astype(np.uint8)).permute(2, 0, 1).float()
            frame = frame[None].to(self.device)
            frame_list.append(frame)
        
        if not frame_list:
            raise ValueError(f"文件夹中没有有效图片: {img_folder}")
        
        return frame_list


def evaluate_video_folder(folder_path, model_path, output_path=None, 
                          device=None, save_flow_scores=False, raft_path=None):
    """
    评估文件夹中所有视频的动态程度
    
    Args:
        folder_path: 视频文件夹路径
        model_path: RAFT 模型权重路径
        output_path: 输出 JSON 文件路径
        device: 计算设备
        save_flow_scores: 是否保存详细的光流分数
        raft_path: RAFT 仓库路径
    """
    # 设置 RAFT 路径
    setup_raft_path(raft_path)
    
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    print(f"使用设备: {device}")
    
    # 初始化模型
    args = edict({
        "model": model_path,
        "small": False,
        "mixed_precision": False,
        "alternate_corr": False
    })
    
    evaluator = DynamicDegree(args, device)
    
    # 获取所有视频文件
    video_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]
    video_files.sort()
    
    if not video_files:
        print(f"文件夹 {folder_path} 中没有找到视频文件")
        return None
    
    print(f"找到 {len(video_files)} 个视频文件")
    print("-" * 70)
    
    results = []
    
    for video_file in tqdm(video_files, desc="评估动态程度"):
        video_path = os.path.join(folder_path, video_file)
        prompt = os.path.splitext(video_file)[0]
        
        try:
            eval_result = evaluator.infer(video_path)
            
            # 计算归一化分数 (0-100)
            dynamic_score = min(100, (eval_result['mean_flow_score'] / 30.0) * 100)
            
            result = {
                'video_file': video_file,
                'prompt': prompt,
                'video_path': video_path,
                'dynamic_score': round(dynamic_score, 2),
                'is_dynamic': eval_result['is_dynamic'],
                'mean_flow_score': round(eval_result['mean_flow_score'], 4),
                'max_flow_score': round(eval_result['max_flow_score'], 4),
                'min_flow_score': round(eval_result['min_flow_score'], 4),
                'std_flow_score': round(eval_result['std_flow_score'], 4),
                'threshold': round(eval_result['threshold'], 4),
                'count_threshold': eval_result['count_threshold'],
                'num_frames': eval_result['num_frames']
            }
            
            if save_flow_scores:
                result['flow_scores'] = [round(s, 4) for s in eval_result['flow_scores']]
            
            results.append(result)
            
        except Exception as e:
            print(f"\n⚠ 评估 {video_file} 时出错: {e}")
            results.append({
                'video_file': video_file,
                'prompt': prompt,
                'video_path': video_path,
                'error': str(e)
            })
    
    # 统计结果
    valid_results = [r for r in results if 'dynamic_score' in r]
    
    if valid_results:
        scores = [r['dynamic_score'] for r in valid_results]
        mean_flow_scores = [r['mean_flow_score'] for r in valid_results]
        dynamic_count = sum(1 for r in valid_results if r['is_dynamic'])
        
        summary = {
            'evaluation_time': datetime.now().isoformat(),
            'folder_path': os.path.abspath(folder_path),
            'model_path': model_path,
            'device': str(device),
            'total_videos': len(video_files),
            'evaluated_videos': len(valid_results),
            'failed_videos': len(video_files) - len(valid_results),
            'dynamic_videos': dynamic_count,
            'static_videos': len(valid_results) - dynamic_count,
            'dynamic_ratio': round(dynamic_count / len(valid_results) * 100, 2),
            'avg_dynamic_score': round(np.mean(scores), 2),
            'max_dynamic_score': round(np.max(scores), 2),
            'min_dynamic_score': round(np.min(scores), 2),
            'std_dynamic_score': round(np.std(scores), 2),
            'avg_mean_flow': round(np.mean(mean_flow_scores), 4),
        }
        
        # 打印结果
        print("\n" + "=" * 70)
        print("📊 评估结果汇总 (RAFT 模型)")
        print("=" * 70)
        print(f"总视频数:       {summary['total_videos']}")
        print(f"成功评估:       {summary['evaluated_videos']}")
        print(f"评估失败:       {summary['failed_videos']}")
        print("-" * 70)
        print(f"动态视频:       {summary['dynamic_videos']} ({summary['dynamic_ratio']:.1f}%)")
        print(f"静态视频:       {summary['static_videos']} ({100 - summary['dynamic_ratio']:.1f}%)")
        print("-" * 70)
        print(f"平均动态分数:   {summary['avg_dynamic_score']:.2f} / 100")
        print(f"最高动态分数:   {summary['max_dynamic_score']:.2f} / 100")
        print(f"最低动态分数:   {summary['min_dynamic_score']:.2f} / 100")
        print(f"分数标准差:     {summary['std_dynamic_score']:.2f}")
        print(f"平均光流幅度:   {summary['avg_mean_flow']:.4f}")
        print("=" * 70)
        
        # 打印详细结果
        print("\n📋 详细结果 (按动态分数降序排列):")
        print("-" * 80)
        print(f"{'状态':<6} | {'分数':>6} | {'光流均值':>8} | {'帧数':>4} | {'Prompt':<40}")
        print("-" * 80)
        
        sorted_results = sorted(valid_results, key=lambda x: x['dynamic_score'], reverse=True)
        for r in sorted_results:
            status = "✓ 动态" if r['is_dynamic'] else "✗ 静态"
            prompt_display = r['prompt'][:38] + '..' if len(r['prompt']) > 40 else r['prompt']
            print(f"{status:<6} | {r['dynamic_score']:>6.2f} | {r['mean_flow_score']:>8.2f} | {r['num_frames']:>4} | {prompt_display}")
        
        print("-" * 80)
    else:
        summary = {
            'evaluation_time': datetime.now().isoformat(),
            'folder_path': os.path.abspath(folder_path),
            'model_path': model_path,
            'total_videos': len(video_files),
            'evaluated_videos': 0,
            'error': 'No videos were successfully evaluated'
        }
    
    # 保存结果
    if output_path is None:
        output_path = os.path.join(folder_path, 'dynamic_degree_results.json')
    
    output_data = {
        'summary': summary,
        'results': results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 结果已保存到: {output_path}")
    
    # 保存 CSV
    csv_path = output_path.replace('.json', '.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("video_file,prompt,dynamic_score,is_dynamic,mean_flow_score,max_flow_score,num_frames\n")
        for r in results:
            if 'dynamic_score' in r:
                prompt_escaped = r['prompt'].replace('"', '""')
                f.write(f'"{r["video_file"]}","{prompt_escaped}",{r["dynamic_score"]},{r["is_dynamic"]},{r["mean_flow_score"]},{r["max_flow_score"]},{r["num_frames"]}\n')
    
    print(f"📄 CSV 结果已保存到: {csv_path}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='使用 RAFT 模型评估视频动态程度',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python dynamic_degree_raft.py /path/to/videos --model models/raft-things.pth
  python dynamic_degree_raft.py /path/to/videos --model models/raft-sintel.pth -o results.json
  python dynamic_degree_raft.py /path/to/videos --model models/raft-things.pth --device cuda:0

准备工作:
  1. 克隆 RAFT 仓库:
     git clone https://github.com/princeton-vl/RAFT.git
  
  2. 下载预训练模型:
     cd RAFT
     wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
     unzip models.zip
  
  3. 安装依赖:
     pip install torch torchvision opencv-python numpy tqdm easydict
        """
    )
    
    parser.add_argument(
        'folder',
        type=str,
        help='包含视频文件的文件夹路径'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='RAFT 模型权重路径 (如: models/raft-things.pth)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出 JSON 文件路径'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='计算设备 (如: cuda:0, cpu)'
    )
    parser.add_argument(
        '--raft_path',
        type=str,
        default="./VBench/vbench/third_party/RAFT",
        help='RAFT 仓库路径'
    )
    parser.add_argument(
        '--save_flow_scores',
        action='store_true',
        help='是否保存每帧的光流分数'
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.folder):
        print(f"❌ 错误: {args.folder} 不是有效的文件夹路径")
        return
    
    if not os.path.exists(args.model):
        print(f"❌ 错误: 模型文件不存在: {args.model}")
        print("请下载 RAFT 预训练模型")
        return
    
    evaluate_video_folder(
        folder_path=args.folder,
        model_path=args.model,
        output_path=args.output,
        device=args.device,
        save_flow_scores=args.save_flow_scores,
        raft_path=args.raft_path
    )


if __name__ == '__main__':
    main()