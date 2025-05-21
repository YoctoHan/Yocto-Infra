#include <torch/torch.h>
#include <torch/extension.h>

#define CUDA_CHECK(status)                                                  \
{                                                                           \
    cudaError_t error = status;                                             \
    if (error != cudaSuccess) {                                             \
        std::cerr << "Got bad cuda status: " << cudaGetErrorString(error)   \
                    << "at line: " << __LINE__ << std::endl;                \
        exit(EXIT_FAILURE);                                                 \
    }                                                                       \
}                                                            \

__global__ void scaled_dot_product_attention(float* Q  , float* K, float* V, float* output, 
                                       int batch_size, int head_num, int seq_len, int head_embd) {
    // 计算当前线程处理的位置
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / (head_num * seq_len);
    int h = (idx / seq_len) % head_num;
    int s = idx % seq_len;
    
    if (idx >= batch_size * head_num * seq_len) {
        return;
    }

    // 计算基础偏移
    int base_offset = (b * head_num + h) * seq_len * head_embd;
    int q_offset = base_offset + s * head_embd;
    
    // 初始化累积值
    float m_prev = -INFINITY;  // 之前的最大值
    float d_prev = 0.0f;       // 之前的分母累积值
    
    // 使用共享内存仅存储临时输出向量
    extern __shared__ float shared_mem[];
    float* o_prev = &shared_mem[threadIdx.x * head_embd];
    
    // 初始化输出向量
    for (int i = 0; i < head_embd; i++) {
        o_prev[i] = 0.0f;
    }
    
    // 单次遍历实现注意力计算
    for (int j = 0; j < seq_len; j++) {
        // 计算点积 x_i
        float dot_product = 0.0f;
        int k_offset = base_offset + j * head_embd;
        int v_offset = base_offset + j * head_embd;
        
        // 直接从全局内存计算点积
        for (int i = 0; i < head_embd; i++) {
            dot_product += Q[q_offset + i] * K[k_offset + i];
        }
        float x_i = dot_product / sqrtf((float)head_embd);
        
        // 2. 更新最大值 m_i
        float m_i = fmaxf(m_prev, x_i);
        
        // 3. 更新分母 d'_i
        float d_i = d_prev * expf(m_prev - m_i) + expf(x_i - m_i);
        
        // 4. 更新输出向量 o'_i
        float scale_prev = d_prev * expf(m_prev - m_i) / d_i;
        float scale_curr = expf(x_i - m_i) / d_i;
        
        // 直接从全局内存读取V值并更新输出
        for (int i = 0; i < head_embd; i++) {
            o_prev[i] = o_prev[i] * scale_prev + scale_curr * V[v_offset + i];
        }
        
        // 更新累积值，为下一轮迭代做准备
        m_prev = m_i;
        d_prev = d_i;
    }
    
    // 将最终结果写入输出数组
    for (int i = 0; i < head_embd; i++) {
        output[q_offset + i] = o_prev[i];
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
        
        // 计算共享内存大小：仅为每个线程的输出向量分配共享内存
        int shared_memory_size = threadsPerBlock * head_embd * sizeof(float);
        
        // 获取并输出共享内存信息
        int max_shared_mem = 0;
        int max_threads_per_block = 0;
        int compute_capability_major = 0;
        int compute_capability_minor = 0;

        cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        cudaDeviceGetAttribute(&max_threads_per_block, cudaDevAttrMaxThreadsPerBlock, 0);
        cudaDeviceGetAttribute(&compute_capability_major, cudaDevAttrComputeCapabilityMajor, 0);
        cudaDeviceGetAttribute(&compute_capability_minor, cudaDevAttrComputeCapabilityMinor, 0);

        std::cout << "\n--- 共享内存信息 ---" << std::endl;
        std::cout << "设备最大共享内存: " << max_shared_mem / 1024.0 << " KB" << std::endl;
        std::cout << "请求的共享内存: " << shared_memory_size / 1024.0 << " KB" << std::endl;
        std::cout << "每个线程共享内存: " << (float)head_embd * sizeof(float) << " bytes" << std::endl;
        std::cout << "线程数量: " << threadsPerBlock << " (最大支持: " << max_threads_per_block << ")" << std::endl;
        std::cout << "CUDA计算能力: " << compute_capability_major << "." << compute_capability_minor << std::endl;

        // 检查共享内存大小
        if (shared_memory_size > max_shared_mem) {
            std::stringstream error_msg;
            error_msg << "请求的共享内存大小 (" << shared_memory_size / 1024.0 
                    << " KB) 超过设备限制 (" << max_shared_mem / 1024.0 << " KB)";
            throw std::runtime_error(error_msg.str());
        }
        
        scaled_dot_product_attention<<<blocks, threadsPerBlock, shared_memory_size>>>(
            Q.data_ptr<float>(), 
            K.data_ptr<float>(), 
            V.data_ptr<float>(), 
            output.data_ptr<float>(),
            batch_size,
            head_num,
            seq_len,
            head_embd);

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