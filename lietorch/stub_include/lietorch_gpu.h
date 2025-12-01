#ifndef LIETORCH_GPU_H_
#define LIETORCH_GPU_H_

#include <vector>
#include <torch/extension.h>

// CPU-only stub implementations. Any call will raise at runtime.

inline torch::Tensor exp_forward_gpu(int, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}
inline std::vector<torch::Tensor> exp_backward_gpu(int, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor log_forward_gpu(int, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}
inline std::vector<torch::Tensor> log_backward_gpu(int, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor inv_forward_gpu(int, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}
inline std::vector<torch::Tensor> inv_backward_gpu(int, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor mul_forward_gpu(int, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}
inline std::vector<torch::Tensor> mul_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor adj_forward_gpu(int, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}
inline std::vector<torch::Tensor> adj_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor adjT_forward_gpu(int, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}
inline std::vector<torch::Tensor> adjT_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor act_forward_gpu(int, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}
inline std::vector<torch::Tensor> act_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor act4_forward_gpu(int, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}
inline std::vector<torch::Tensor> act4_backward_gpu(int, torch::Tensor, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor orthogonal_projector_gpu(int, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor as_matrix_forward_gpu(int, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

inline torch::Tensor jleft_forward_gpu(int, torch::Tensor, torch::Tensor) {
    TORCH_CHECK(false, "lietorch was built without CUDA support");
}

#endif // LIETORCH_GPU_H_
