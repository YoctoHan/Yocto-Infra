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

__global__ void scaled_dot_product_attention(float* Q, float* K, float* V, float* output, 
                                       int batch_size, int head_num, int seq_len, int head_embd) {
    // 计算当前线程处理的位置
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算当前批次和注意力头
    int b = idx / (head_num * seq_len);
    int h = (idx / seq_len) % head_num;
    int s = idx % seq_len;
    
    if (idx >= batch_size * head_num * seq_len) {
        return;
    }

    float max_value = -INFINITY;
    float denominator = 0.0f;
    
    // 使用共享内存动态分配
    extern __shared__ float shared_memory[];
    float* scaled_dot_product_result = shared_memory;
    
    // 计算当前位置的基础偏移
    int base_offset = (b * head_num + h) * seq_len * head_embd;
    
    for (int idy = 0; idy < seq_len; idy++) {
        float sum = 0.0f;
        for (int i = 0; i < head_embd; i++) {
            sum += Q[base_offset + s * head_embd + i] * K[base_offset + idy * head_embd + i];
        }
        float weight = sum / sqrtf((float)head_embd);
        scaled_dot_product_result[threadIdx.x * seq_len + idy] = weight;
        max_value = fmax(max_value, weight);
    }

    for (int i = 0; i < seq_len; i++) {
        float exp_value = expf(scaled_dot_product_result[threadIdx.x * seq_len + i] - max_value);
        denominator += exp_value;
        scaled_dot_product_result[threadIdx.x * seq_len + i] = exp_value;
    }

    for (int i = 0; i < seq_len; i++) {
        scaled_dot_product_result[threadIdx.x * seq_len + i] /= denominator;
    }

    for (int idy = 0; idy < head_embd; idy++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += scaled_dot_product_result[threadIdx.x * seq_len + j] * V[base_offset + j * head_embd + idy];
        }
        output[base_offset + s * head_embd + idy] = sum;
    }
}

// 方法二：使用全局内存实现
__global__ void scaled_dot_product_attention_global_mem(float* Q, float* K, float* V, float* output, 
                                                       float* temp_storage, 
                                                       int batch_size, int head_num, int seq_len, int head_embd) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 计算当前批次和注意力头
    int b = idx / (head_num * seq_len);
    int h = (idx / seq_len) % head_num;
    int s = idx % seq_len;
    
    if (idx >= batch_size * head_num * seq_len) {
        return;
    }

    float max_value = -INFINITY;
    float denominator = 0.0f;
    
    // 使用全局内存
    float* scaled_dot_product_result = temp_storage + idx * seq_len;
    
    // 计算当前位置的基础偏移
    int base_offset = (b * head_num + h) * seq_len * head_embd;

    for (int idy = 0; idy < seq_len; idy++) {
        float sum = 0.0f;
        for (int i = 0; i < head_embd; i++) {
            sum += Q[base_offset + s * head_embd + i] * K[base_offset + idy * head_embd + i];
        }
        float weight = sum / sqrtf((float)head_embd);
        scaled_dot_product_result[idy] = weight;
        max_value = fmax(max_value, weight);
    }

    for (int i = 0; i < seq_len; i++) {
        float exp_value = expf(scaled_dot_product_result[i] - max_value);
        denominator += exp_value;
        scaled_dot_product_result[i] = exp_value;
    }

    for (int i = 0; i < seq_len; i++) {
        scaled_dot_product_result[i] /= denominator;
    }

    for (int idy = 0; idy < head_embd; idy++) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += scaled_dot_product_result[j] * V[base_offset + j * head_embd + idy];
        }
        output[base_offset + s * head_embd + idy] = sum;
    }
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 检查输入张量维度是否为4维 [batch_size, head_num, seq_len, head_embd]
    if (Q.dim() != 4 || K.dim() != 4 || V.dim() != 4) {
        throw std::runtime_error("输入张量必须是4维的 [batch_size, head_num, seq_len, head_embd]");
    }
    
    // 检查维度匹配
    if (Q.size(0) != K.size(0) || Q.size(0) != V.size(0) ||  // batch_size
        Q.size(1) != K.size(1) || Q.size(1) != V.size(1) ||  // head_num
        Q.size(2) != K.size(2) || Q.size(2) != V.size(2) ||  // seq_len
        Q.size(3) != K.size(3) || Q.size(3) != V.size(3)) {  // head_embd
        throw std::runtime_error("输入张量尺寸不匹配");
    }
    
    // 获取各个维度大小
    int batch_size = Q.size(0);
    int head_num = Q.size(1);
    int seq_len = Q.size(2);
    int head_embd = Q.size(3);

    if (Q.device() != K.device() || K.device() != V.device()) {
        throw std::runtime_error("所有输入张量必须在相同设备上");
    }

    if (!Q.is_cuda()) {
        throw std::runtime_error("输入张量必须在CUDA设备上");
    }

    if (Q.scalar_type() != torch::kFloat32) {
        throw std::runtime_error("输入张量必须是float32类型");
    }

    torch::Tensor output = torch::zeros_like(Q);

    try {
        int threadsPerBlock = 256;
        int total_sequences = batch_size * head_num * seq_len;
        int blocks = (total_sequences + threadsPerBlock - 1) / threadsPerBlock;
        
        // 方法一：使用共享内存
        int shared_memory_size = threadsPerBlock * seq_len * sizeof(float);
        
        // 判断是否超过共享内存限制（一般为48KB或96KB）
        if (shared_memory_size <= 49152) { // 48KB
            scaled_dot_product_attention<<<blocks, threadsPerBlock, shared_memory_size>>>(
                Q.data_ptr<float>(), 
                K.data_ptr<float>(), 
                V.data_ptr<float>(), 
                output.data_ptr<float>(),
                batch_size,
                head_num,
                seq_len,
                head_embd);
        } else {
            // 方法二：超过共享内存限制时使用全局内存
            torch::Tensor temp_storage = torch::empty({total_sequences * seq_len}, 
                                                     torch::TensorOptions().device(Q.device()).dtype(torch::kFloat32));
            
            scaled_dot_product_attention_global_mem<<<blocks, threadsPerBlock>>>(
                Q.data_ptr<float>(), 
                K.data_ptr<float>(), 
                V.data_ptr<float>(), 
                output.data_ptr<float>(),
                temp_storage.data_ptr<float>(),
                batch_size,
                head_num,
                seq_len,
                head_embd);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        return output;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("CUDA执行错误: ") + e.what());
    }
}

int main() {
    try {
        torch::manual_seed(12333);

        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA 不可用, 请检查 CUDA 环境.");
        }

        // 定义四维输入的维度
        int batch_size = 2;
        int head_num = 8;
        int seq_len = 512; // 可变
        int head_embd = 64; // 可变
        
        std::cout << "使用批次大小: " << batch_size << ", 注意力头数: " << head_num 
                  << ", 序列长度: " << seq_len << ", 头嵌入维度: " << head_embd << std::endl;

        // 创建四维输入
        torch::Tensor Q = torch::randn({batch_size, head_num, seq_len, head_embd});
        torch::Tensor K = torch::randn({batch_size, head_num, seq_len, head_embd});
        torch::Tensor V = torch::randn({batch_size, head_num, seq_len, head_embd});

        std::cout << "Q.shape = " << Q.sizes() << std::endl;
        std::cout << "K.shape = " << K.sizes() << std::endl;
        std::cout << "V.shape = " << V.sizes() << std::endl;

        Q = Q.cuda();
        K = K.cuda();
        V = V.cuda();

        torch::Tensor output = forward(Q, K, V);
        
        std::cout << "注意力计算完成" << std::endl;
        std::cout << "输出张量形状: " << output.sizes() << std::endl;
        
        std::cout << "输出的平均值: " << output.mean().item<float>() << std::endl;
        std::cout << "输出的标准差: " << output.std().item<float>() << std::endl;
    } catch (std::exception& e) {
        std::cout << "错误: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}