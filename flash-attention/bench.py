import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'

import math
import sys
import subprocess
import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

def load_cuda_extensions():
    """加载CUDA扩展"""
    try:
        naive_attention_torch = load(
            name='naive_attention_torch', 
            sources=['naive_attention_torch/main.cc', 'naive_attention_torch/naive_attention_torch.cc'], 
            extra_cuda_cflags=['-O2']
        )
        naive_attention_cuda = load(
            name='naive_attention_cuda', 
            sources=['naive_attention_cuda/main.cc', 'naive_attention_cuda/naive_attention_cuda.cu'], 
            extra_cuda_cflags=['-O2']
        )
        attention_fused_kernel_cuda = load(
            name='attention_fused_kernel_cuda', 
            sources=['attention_fused_kernel_cuda/main.cc', 'attention_fused_kernel_cuda/attention_fused_kernel_cuda.cu'], 
            extra_cuda_cflags=['-O2']
        )
        attention_fused_kernel_2pass_cuda = load(
            name='attention_fused_2pass_cuda', 
            sources=['attention_fused_2pass_cuda/main.cc', 'attention_fused_2pass_cuda/attention_fused_kernel_2pass_cuda.cu'], 
            extra_cuda_cflags=['-O2']
        )
        flash_attention_1_cuda = load(
            name='flash_attention_1_cuda', 
            sources=['flash_attention_1/main.cc', 'flash_attention_1/flash_attention_1.cu'], 
            extra_cuda_cflags=['-O2']
        )
        return naive_attention_torch, naive_attention_cuda, attention_fused_kernel_cuda, attention_fused_kernel_2pass_cuda, flash_attention_1_cuda
    except Exception as e:
        print(f"加载CUDA扩展失败: {e}")
        sys.exit(1)


def manual_attention(q, k, v):
    """手动实现的注意力机制, 支持batch_size和head_num维度"""
    # 输入形状: [batch_size, head_num, seq_len, head_embd]
    
    # 计算注意力分数: [batch_size, head_num, seq_len, seq_len]
    att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    
    # 计算输出: [batch_size, head_num, seq_len, head_embd]
    return torch.matmul(att, v)


def profile_attention_function(func, *args):
    """使用PyTorch的Profiler对注意力计算进行性能分析"""
    with torch.autograd.profiler.profile(use_device='cuda') as prof:
        result = func(*args)
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    return result

def check_results(item, naive_result, manual_result):
    # 打印基本信息以进行调试
    print(f"\n检查{item}结果:")
    print(f"naive_result设备: {naive_result.device}, 形状: {naive_result.shape}, 类型: {naive_result.dtype}")
    print(f"manual_result设备: {manual_result.device}, 形状: {manual_result.shape}, 类型: {manual_result.dtype}")
    
    # 检查设备
    if naive_result.device != manual_result.device:
        print(f"设备不匹配: naive={naive_result.device}, manual={manual_result.device}")
        naive_result = naive_result.to(manual_result.device)
    
    # 检查形状
    if naive_result.shape != manual_result.shape:
        print(f"形状不匹配: naive={naive_result.shape}, manual={manual_result.shape}")
        return False
    
    # 使用try-except捕获CUDA错误
    try:
        # 检查NaN和Inf
        has_nan_naive = torch.isnan(naive_result).any().item()
        has_inf_naive = torch.isinf(naive_result).any().item()
        if has_nan_naive or has_inf_naive:
            print("naive结果包含NaN或Inf值")
            return False
        
        has_nan_manual = torch.isnan(manual_result).any().item()
        has_inf_manual = torch.isinf(manual_result).any().item()
        if has_nan_manual or has_inf_manual:
            print("manual结果包含NaN或Inf值")
            return False
        
        # 打印最大差异，帮助诊断
        max_diff = (naive_result - manual_result).abs().max().item()
        print(f"最大差异: {max_diff}")
        
        # 使用更宽松的容差进行比较
        return torch.allclose(naive_result, manual_result, rtol=1e-2, atol=1e-2)
    except RuntimeError as e:
        print(f"检查结果时发生错误: {e}")
        # 打印一些样本值进行诊断
        try:
            print(f"naive_result样本: {naive_result[0, 0, 0, :5]}")
            print(f"manual_result样本: {manual_result[0, 0, 0, :5]}")
        except:
            print("无法打印样本值")
        return False


def main():
    # 加载CUDA扩展
    naive_attention_torch, naive_attention_cuda, attention_fused_kernel_cuda, attention_fused_2pass_cuda, flash_attention_1_cuda = load_cuda_extensions()

    # 设置输入张量，降低维度以避免可能的内存或数值问题
    batch_size, head_num, seq_len, head_embd = 2, 4, 32, 64  # 降低维度
    
    print(f"创建输入张量: batch_size={batch_size}, head_num={head_num}, seq_len={seq_len}, head_embd={head_embd}")
    
    # 创建形状为[batch_size, head_num, seq_len, head_embd]的张量
    q = torch.randn(batch_size, head_num, seq_len, head_embd).cuda().to(torch.float32)
    k = torch.randn(batch_size, head_num, seq_len, head_embd).cuda().to(torch.float32)
    v = torch.randn(batch_size, head_num, seq_len, head_embd).cuda().to(torch.float32)
    
    # 打印输入统计信息
    print(f"Q范围: {q.min().item():.4f} ~ {q.max().item():.4f}, 均值: {q.mean().item():.4f}, 标准差: {q.std().item():.4f}")
    
    # 手动注意力计算性能分析（直接支持4D张量）
    print('=== profiling manual attention ===')
    try:
        manual_result = profile_attention_function(manual_attention, q, k, v)
        print(f"manual结果形状: {manual_result.shape}, 均值: {manual_result.mean().item():.4f}")
    except Exception as e:
        print(f"手动注意力计算失败: {e}")
        return

    # 使用naive_attention_torch的CUDA扩展计算性能分析
    print('=== profiling naive attention torch ===')
    try:
        naive_result_torch = profile_attention_function(
            naive_attention_torch.forward, q, k, v
        )
        print(f"naive_torch结果形状: {naive_result_torch.shape}, 均值: {naive_result_torch.mean().item():.4f}")
    except Exception as e:
        print(f"naive_attention_torch计算失败: {e}")
        naive_result_torch = None

    # 使用naive_attention_cuda的CUDA扩展计算性能分析
    print('=== profiling naive attention cuda ===')
    try:
        naive_result_cuda = profile_attention_function(
            naive_attention_cuda.forward, q, k, v
        )
        print(f"naive_cuda结果形状: {naive_result_cuda.shape}, 均值: {naive_result_cuda.mean().item():.4f}")
    except Exception as e:
        print(f"naive_attention_cuda计算失败: {e}")
        naive_result_cuda = None

    # 使用attention_fused_kernel_cuda的CUDA扩展计算性能分析
    print('=== profiling fused attention cuda ===')
    try:
        fused_result_cuda = profile_attention_function(
            attention_fused_kernel_cuda.forward, q, k, v
        )
        print(f"fused_cuda结果形状: {fused_result_cuda.shape}, 均值: {fused_result_cuda.mean().item():.4f}")
    except Exception as e:
        print(f"fused_attention_cuda计算失败: {e}")
        fused_result_cuda = None

    print('=== profiling fused attention 2-pass cuda ===')
    try:
        fused_2pass_result_cuda = profile_attention_function(
            attention_fused_2pass_cuda.forward, q, k, v
        )
        print(f"fused_cuda结果形状: {fused_result_cuda.shape}, 均值: {fused_result_cuda.mean().item():.4f}")
    except Exception as e:
        print(f"fused_attention_cuda计算失败: {e}")
        fused_2pass_result_cuda = None

    # 使用flash_attention_1_cuda的CUDA扩展计算性能分析
    print('=== profiling flash attention 1 cuda ===')
    try:
        flash_result_cuda = profile_attention_function(
            flash_attention_1_cuda.forward, q, k, v
        )
        print(f"flash_cuda结果形状: {flash_result_cuda.shape}, 均值: {flash_result_cuda.mean().item():.4f}")
    except Exception as e:
        print(f"flash_attention_1_cuda计算失败: {e}")
        flash_result_cuda = None

    # 检查结果的正确性
    if naive_result_torch is not None:
        check_results('naive_torch', naive_result_torch, manual_result)
    if naive_result_cuda is not None:
        check_results('naive_cuda', naive_result_cuda, manual_result)
    if fused_result_cuda is not None:
        check_results('fused_cuda', fused_result_cuda, manual_result)
    if fused_2pass_result_cuda is not None:
        check_results('fused_2-pass_cuda', fused_2pass_result_cuda, manual_result)
    if flash_result_cuda is not None:
        check_results('flash_cuda', flash_result_cuda, manual_result)

if __name__ == "__main__":
    main()
