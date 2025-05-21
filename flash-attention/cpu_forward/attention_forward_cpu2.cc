#ifdef HAVE_TORCH
#include <torch/types.h>
#endif

#include <random>
#include <vector>
#include <cassert>
#include <iostream>
#include <cmath> // 添加数学函数头文件

std::vector<float> softmax_naive(const std::vector<float>& vector) { // 添加引用
    const int length = vector.size();
    assert(length > 0);

    float denominator = 0.0f;
    for (auto& element : vector) {
        denominator += std::exp(element); // 添加std命名空间
    }

    std::vector<float> output(length);
    for (int i = 0; i < length; i++) {
        output[i] = std::exp(vector[i]) / denominator; // 修正softmax计算
    }

    return output;
}

std::vector<std::vector<float>> softmax(const std::vector<std::vector<float>>& mat) {
    const int row = mat.size();
    assert(row > 0);
    const int column = mat[0].size();
    assert(column > 0);

    std::vector<std::vector<float>> output;
    for (auto& vector : mat) {
        output.emplace_back(softmax_naive(vector));
    }
    return output;
}

void scale(std::vector<std::vector<float>>& mat, const float scale) {
    const int row = mat.size();
    assert(row > 0);
    const int column = mat[0].size();
    assert(column > 0);
    assert(scale != 0.0f);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            mat[i][j] /= scale;
        }
    }
}

std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& mat) {
    const int row = mat.size();
    assert(row > 0);
    const int column = mat[0].size();
    assert(column > 0);
    
    std::vector<std::vector<float>> output(column, std::vector<float>(row));
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            output[j][i] = mat[i][j];
        }
    }
    return output;
}

std::vector<std::vector<float>> mat_mul(const std::vector<std::vector<float>>& mat_1,
                                        const std::vector<std::vector<float>>& mat_2) {
    const int row = mat_1.size();
    assert(row > 0);
    const int public_dim = mat_1[0].size();
    assert(public_dim > 0);
    assert(public_dim == mat_2.size());
    const int column = mat_2[0].size();
    assert(column > 0);

    std::vector<std::vector<float>> output(row, std::vector<float>(column)); // 修正大小
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            for (int k = 0; k < public_dim; k++) {
                output[i][j] += mat_1[i][k] * mat_2[k][j];
            }
        }
    }
    return output;
}

std::vector<std::vector<float>> attention(const std::vector<std::vector<float>>& Q,
                                          const std::vector<std::vector<float>>& K,
                                          const std::vector<std::vector<float>>& V) {
    std::vector<std::vector<float>> output;
    std::vector<std::vector<float>> mat_mul_output;
    mat_mul_output = mat_mul(Q, transpose(K));
    scale(mat_mul_output, std::sqrt(K[0].size())); // 使用std::sqrt替代fsqrt
    output = softmax(mat_mul_output); // 修正参数
    output = mat_mul(output, V);

    return output;
}

void initial_mat2d(std::vector<std::vector<float>>& mat, std::mt19937& gen, std::uniform_real_distribution<float>& dist) {
    const int row = mat.size();
    assert(row > 0);
    const int column = mat[0].size();
    assert(column > 0);

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < column; j++) {
            mat[i][j] = dist(gen);
        }
    }
}

#ifdef HAVE_TORCH
// PyTorch接口函数
torch::Tensor torch_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 这里实现PyTorch张量的处理逻辑
    // 此处提供一个简单的实现框架，实际使用时需根据具体情况调整
    
    // 假设张量是2D的 [seq_len, feature_dim]
    int seq_len = Q.size(0);
    int feature_dim = Q.size(1);
    
    // 创建返回值张量
    auto options = torch::TensorOptions().dtype(Q.dtype()).device(Q.device());
    torch::Tensor output = torch::zeros({seq_len, feature_dim}, options);
    
    // 在这里可以插入实际的attention逻辑
    // 如果需要使用上面的函数，需要先将PyTorch张量转换为std::vector
    
    return output;
}
#endif

// 主函数用于CPU版本的attention演示
int main() {
    int seq_len = 10;
    int feature_dim = 128;

    // 初始化随机数生成器
    std::random_device random_device;
    std::mt19937 gen(random_device());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 初始化 Q、K 和 V 矩阵
    std::vector<std::vector<float>> Q_mat(seq_len, std::vector<float>(feature_dim));
    std::vector<std::vector<float>> K_mat(seq_len, std::vector<float>(feature_dim));
    std::vector<std::vector<float>> V_mat(seq_len, std::vector<float>(feature_dim));

    // 使用随机数填充矩阵
    initial_mat2d(Q_mat, gen, dist);
    initial_mat2d(K_mat, gen, dist);
    initial_mat2d(V_mat, gen, dist);

    std::vector<std::vector<float>> output = attention(Q_mat, K_mat, V_mat);

    // 输出结果（可选）
    std::cout << "Attention 输出的第一行前几个元素:" << std::endl;
    if (!output.empty() && !output[0].empty()) {
        for (int i = 0; i < std::min(5, (int)output[0].size()); i++) {
            std::cout << output[0][i] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
} 