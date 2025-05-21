#include <torch/torch.h>

torch::Tensor softmax_naive(const torch::Tensor& tensor) {
    int length = tensor.size(-1);
    assert(length > 0);

    // 确保输入没有NaN
    if (tensor.isnan().any().item<bool>()) {
        throw std::runtime_error("Input tensor contains NaN values");
    }

    // 数值稳定性处理
    torch::Tensor max_val = std::get<0>(tensor.max(-1, true));
    torch::Tensor shifted = tensor - max_val;
    torch::Tensor exp_tensor = shifted.exp();
    torch::Tensor sum_exp = exp_tensor.sum(-1, true);
    
    // 防止除零
    sum_exp = sum_exp.clamp_min(1e-10);
    
    return exp_tensor / sum_exp;
}

torch::Tensor softmax(const torch::Tensor& mat) {
    // 支持四维张量 [batch_size, head_num, seq_len, seq_len]
    assert(mat.dim() == 4);
    int batch_size = mat.size(0);
    int head_num = mat.size(1);
    int seq_len = mat.size(2);
    
    torch::Tensor output = torch::zeros_like(mat);
    for (int b = 0; b < batch_size; b++) {
        for (int h = 0; h < head_num; h++) {
            for (int i = 0; i < seq_len; i++) {
                output[b][h][i] = softmax_naive(mat[b][h][i]);
            }
        }
    }
    return output;
}

torch::Tensor mat_mul(const torch::Tensor& mat_1, const torch::Tensor& mat_2) {
    // 支持批量矩阵乘法，使用torch内置函数
    return torch::matmul(mat_1, mat_2);
}

torch::Tensor attention(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V) {
    // 输入形状: [batch_size, head_num, seq_len, head_embd]
    assert(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4);
    assert(Q.size(0) == K.size(0) && K.size(0) == V.size(0)); // 批次大小相同
    assert(Q.size(1) == K.size(1) && K.size(1) == V.size(1)); // 头数相同
    assert(Q.size(3) == K.size(3)); // Q和K的head_embd维度相同
    assert(K.size(2) == V.size(2)); // K和V的seq_len维度相同
    
    // 转置K的最后两个维度 [batch_size, head_num, seq_len, head_embd] -> [batch_size, head_num, head_embd, seq_len]
    torch::Tensor K_t = K.transpose(-2, -1);
    
    // Q·K^T -> [batch_size, head_num, seq_len_q, seq_len_k]
    torch::Tensor scores = torch::matmul(Q, K_t);
    
    // 缩放
    scores /= std::sqrt(static_cast<float>(K.size(-1)));
    
    // softmax
    torch::Tensor attention_weights = softmax(scores);
    
    // 注意力权重·V -> [batch_size, head_num, seq_len_q, head_embd_v]
    return torch::matmul(attention_weights, V);
}

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 输入形状检查
    assert(Q.dim() == 4 && K.dim() == 4 && V.dim() == 4);
    torch::Tensor output = attention(Q, K, V);
    
    return output;
}
