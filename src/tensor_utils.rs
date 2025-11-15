use burn::tensor::{activation::softmax, backend::Backend, DType, Tensor};

/// Run softmax in fp32 when the current dtype is bf16/f16, casting back afterwards.
pub fn softmax_fp32_if_needed<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    let dtype = tensor.dtype();
    if matches!(dtype, DType::BF16 | DType::F16) {
        softmax(tensor.cast(DType::F32), dim).cast(dtype)
    } else {
        softmax(tensor, dim)
    }
}
