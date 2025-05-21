#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <torch/torch.h>

#define CUDA_CHECK(status)                                                  \
{                                                                           \
    cudaError_t error = status;                                             \
    if (error != cudaSuccess) {                                             \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)   \
                    << "at line: " << __LINE__ << std::endl;                \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}

// CUDA kernel for compute (Q * K^T) / sqrt(feature_dim)
__global__ void compute_attention_weight(float* Q, float* K, float* attention_weight, 
                                        int batch_size, int head_num, int seq_len, int head_embd) {
    // 线程索引计算：每个线程负责一个batch中一个head下的一个序列位置
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_seq = batch_size * head_num * seq_len;
    
    if (idx >= total_seq) {
        return;
    }
    
    // 计算当前线程对应的batch, head和序列位置
    int b = idx / (head_num * seq_len);         // 批次索引
    int h = (idx / seq_len) % head_num;         // 头索引
    int s = idx % seq_len;                      // 序列位置索引
    
    float scale_value = sqrtf(float(head_embd));
    
    // 计算该线程对应的Q和attention_weight的基址偏移
    int q_offset = b * head_num * seq_len * head_embd + h * seq_len * head_embd + s * head_embd;
    int attn_offset = b * head_num * seq_len * seq_len + h * seq_len * seq_len + s * seq_len;
    
    for (int t = 0; t < seq_len; t++) {
        float sum = 0.0f;
        // 计算K对应序列位置t的基址偏移
        int k_offset = b * head_num * seq_len * head_embd + h * seq_len * head_embd + t * head_embd;
        
        for (int i = 0; i < head_embd; i++) {
            sum += Q[q_offset + i] * K[k_offset + i]; // Q[b,h,s,:] * K[b,h,t,:]
        }
        attention_weight[attn_offset + t] = sum / scale_value;
    }
}

// CUDA kernel for apply softmax to attention weight
__global__ void softmax_attention_weight(float* attention_weight, 
                                        int batch_size, int head_num, int seq_len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_seq = batch_size * head_num * seq_len;
    
    if (idx >= total_seq) {
        return;
    }
    
    // 计算当前线程对应的batch, head和序列位置
    int b = idx / (head_num * seq_len);         // 批次索引
    int h = (idx / seq_len) % head_num;         // 头索引
    int s = idx % seq_len;                      // 序列位置索引
    
    // 计算该线程处理的attention权重矩阵行的基址偏移
    int attn_offset = b * head_num * seq_len * seq_len + h * seq_len * seq_len + s * seq_len;
    
    // step 1: compute the max value of each row
    float max_val = attention_weight[attn_offset];
    for (int i = 1; i < seq_len; i++) {
        max_val = fmax(max_val, attention_weight[attn_offset + i]);
    }

    // step 2: compute the denominator of softmax
    float denominator = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        denominator += expf(attention_weight[attn_offset + i] - max_val);
    }

    // step 3: compute the softmax value
    for (int i = 0; i < seq_len; i++) {
        attention_weight[attn_offset + i] = expf(attention_weight[attn_offset + i] - max_val) / denominator;
    }
}

// CUDA kernel to compute the attention output (attention weights * V)
__global__ void apply_attention_to_value(float* attention_weight, float* V, float* output, 
                                         int batch_size, int head_num, int seq_len, int head_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_seq = batch_size * head_num * seq_len;
    
    if (idx >= total_seq) {
        return;
    }
    
    // 计算当前线程对应的batch, head和序列位置
    int b = idx / (head_num * seq_len);         // 批次索引
    int h = (idx / seq_len) % head_num;         // 头索引
    int s = idx % seq_len;                      // 序列位置索引
    
    // 计算该线程处理的attention权重行和输出行的基址偏移
    int attn_offset = b * head_num * seq_len * seq_len + h * seq_len * seq_len + s * seq_len;
    int out_offset = b * head_num * seq_len * head_embd + h * seq_len * head_embd + s * head_embd;
    
    for (int d = 0; d < head_embd; d++) {
        float sum = 0.0f;
        for (int t = 0; t < seq_len; t++) {
            // 计算V对应位置的基址偏移
            int v_offset = b * head_num * seq_len * head_embd + h * seq_len * head_embd + t * head_embd;
            sum += attention_weight[attn_offset + t] * V[v_offset + d];
        }
        output[out_offset + d] = sum;
    }
}


torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 检查输入张量的尺寸 [batch_size, head_num, seq_len, head_embd]
    if (Q.dim() != 4 || K.dim() != 4 || V.dim() != 4) {
        throw std::runtime_error("输入张量必须是4维的 [batch_size, head_num, seq_len, head_embd]");
    }
    
    if (Q.size(0) != K.size(0) || Q.size(0) != V.size(0) || // batch_size
        Q.size(1) != K.size(1) || Q.size(1) != V.size(1) || // head_num
        Q.size(2) != K.size(2) || Q.size(2) != V.size(2) || // seq_len
        Q.size(3) != K.size(3) || K.size(3) != V.size(3)) { // head_embd
        throw std::runtime_error("输入张量的尺寸不匹配");
    }

    int batch_size = Q.size(0);
    int head_num = Q.size(1);
    int seq_len = Q.size(2);
    int head_embd = Q.size(3);

    if (Q.device() != K.device() || K.device() != V.device()) {
        throw std::runtime_error("All input tensor must be on the same device.");
    }

    // 检查设备类型
    if (!Q.is_cuda()) {
        throw std::runtime_error("输入张量必须在CUDA设备上");
    }

    // 检查数据类型
    if (Q.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("输入张量必须是float32类型");
    }

    torch::Tensor output = torch::zeros_like(Q);

    // Allocate memory for the device pointers.
    float *d_attention_weight = nullptr;

    try {
        // 为每个batch和head分配attention权重矩阵
        cudaError_t err = cudaMalloc(&d_attention_weight, batch_size * head_num * seq_len * seq_len * sizeof(float));
        CUDA_CHECK(err);

        // 计算每个batch和head需要的线程数
        int total_threads = batch_size * head_num * seq_len;
        int threadsPerBlock = 256;
        int blocks = (total_threads + threadsPerBlock - 1) / threadsPerBlock;
        
        // stage 1: Compute attention weight (Q * K^T / sqrt(head_embd))
        compute_attention_weight<<<blocks, threadsPerBlock>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), d_attention_weight, 
            batch_size, head_num, seq_len, head_embd);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // stage 2: Apply softmax to attention weight
        softmax_attention_weight<<<blocks, threadsPerBlock>>>(
            d_attention_weight, batch_size, head_num, seq_len);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // stage 3: Apply attention to V (attention_weight * V)
        apply_attention_to_value<<<blocks, threadsPerBlock>>>(
            d_attention_weight, V.data_ptr<float>(), output.data_ptr<float>(), 
            batch_size, head_num, seq_len, head_embd);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } catch (const std::exception& e) {
        // 发生异常时释放资源
        if (d_attention_weight != nullptr) {
            cudaFree(d_attention_weight);
        }
        throw; // 重新抛出异常
    }

    // 释放CUDA内存
    if (d_attention_weight != nullptr) {
        cudaError_t err = cudaFree(d_attention_weight);
        CUDA_CHECK(err);
    }

    return output;
}

int main() {
    try {
        // 设置随机数种子，方便复现结果
        torch::manual_seed(12223);

        // 检查CUDA是否可用
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA不可用, 请检查CUDA环境");
        }

        // 设置输入维度 [batch_size, head_num, seq_len, head_embd]
        int batch_size = 2;
        int head_num = 8;
        int seq_len = 128;
        int head_embd = 64;

        // 使用随机数初始化 Q、K、V 三个张量
        torch::Tensor Q = torch::randn({batch_size, head_num, seq_len, head_embd});
        torch::Tensor K = torch::randn({batch_size, head_num, seq_len, head_embd});
        torch::Tensor V = torch::randn({batch_size, head_num, seq_len, head_embd});

        // 输出 Q、K、V 三个张量的 shape
        std::cout << "Q.shape = " << Q.sizes() << std::endl;
        std::cout << "K.shape = " << K.sizes() << std::endl;
        std::cout << "V.shape = " << V.sizes() << std::endl;

        // 将 Q、K、V 移动到GPU
        Q = Q.cuda();
        K = K.cuda();
        V = V.cuda();

        // 接口调用
        torch::Tensor output = forward(Q, K, V);
        
        std::cout << "注意力计算完成" << std::endl;
        std::cout << "输出张量形状: " << output.sizes() << std::endl;
        
        // 可以进行进一步的结果验证
        std::cout << "输出的平均值: " << output.mean().item<float>() << std::endl;
        std::cout << "输出的标准差: " << output.std().item<float>() << std::endl;
        
    } catch(const std::exception& e) {
        std::cout << "错误:" << e.what() << std::endl;
        return 1;
    }

    return 0;
}