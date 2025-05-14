#include "render.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#include <cub/cub.cuh>

#define SHARED_BATCH_SIZE 128

Render_GPU::Render_GPU()
{
}

Render_GPU::~Render_GPU()
{
}

void Render_GPU::setup(const uint2 dims, const uint2 tile_dims, const uint32_t n_gaussians)
{
}

void Render_GPU::finalize()
{
}

__device__ __forceinline__ float expo(float x) { 
    const float LN2 = 0.69314718056f;
    const float INV_LN2 = 1.44269504089f;

    int n = __float2int_rz(x * INV_LN2);
    float r = x - n * LN2;

    // Horner optimization
    float poly = 1.0f + r * (1.0000026f + r * (0.4999263f + r * (0.1669445f + r * (0.0416618f + r * (0.0083063f + r * (0.0013130f + r * 0.0001887f))))));
    return exp2f((float)n) * poly;
}

__device__ void evaluateGaussianOptimized(const float4 invCov2D_opacity, const float2 mean2D, const float3 color, float3 &outColor, const uint32_t x, const uint32_t y, float &T) {
    if (T < T_THRESHOLD) {
        return;
    }

    const float3 inv_cov2D = make_float3(invCov2D_opacity);
    const float opacity = invCov2D_opacity.w;
    
    const float dx = x - mean2D.x;
    const float dy = y - mean2D.y;
    
    const float power = -0.5f * (inv_cov2D.x * dx * dx + inv_cov2D.y * 2.0f * dx * dy + inv_cov2D.z * dy * dy);
    
    if (power > 0.0f) {
        return;
    }

    float alpha = opacity * expo(power);
    if (alpha < ALPHA_THRESHOLD) {
        return;
    }
    
    alpha = fminf(alpha, 0.99f);
    const float blend_factor = T * alpha;
    
    outColor += color * blend_factor; // small opti
    
    T *= (1.0f - alpha);
}

__global__ void render(const DatasetGPU data, float3 *d_img_out) {
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (!(x < data._dims.x && y < data._dims.y)){
        return;
    }
    
    const uint32_t tile_x = blockIdx.x * blockDim.x / TILE_SIZE;
    const uint32_t tile_y = blockIdx.y * blockDim.y / TILE_SIZE;
    const uint32_t tile_idx = tile_x + tile_y * data._tile_dims.x;
    
    if(!(tile_idx < data._n_tiles)){
        return;
    }
    
    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const uint32_t block_size = blockDim.x * blockDim.y;
    
    uint32_t pix_idx, tile_start, n_gaussians;
    uint32_t batchSize;
    float3 outColor;
    float T;
    
    __shared__ float4 s_invCov2D_opacity[SHARED_BATCH_SIZE];
    __shared__ float2 s_means2D[SHARED_BATCH_SIZE];
    __shared__ float3 s_colors[SHARED_BATCH_SIZE];
    __shared__ uint2 s_tile_range;
    __shared__ uint32_t s_n_gaussians;
    
    if (tid == 0) {
        s_tile_range = data._tile_ranges[tile_idx];
        s_n_gaussians = s_tile_range.y - s_tile_range.x + 1;
    }
    __syncthreads();

    pix_idx = x + y * data._dims.x;
    outColor = make_float3(0.0f);
    T = 1.0f;

    tile_start = s_tile_range.x;
    n_gaussians = s_n_gaussians;

    for(uint32_t batchStart = 0; batchStart < n_gaussians && T >= T_THRESHOLD; batchStart += SHARED_BATCH_SIZE) {
        batchSize = min(SHARED_BATCH_SIZE, n_gaussians - batchStart);

        for(uint32_t i = tid; i < batchSize; i += block_size) {
            uint32_t global_idx = tile_start + batchStart + i;
            uint32_t gaussian_idx = data._tile_splat_entries[global_idx];

            s_invCov2D_opacity[i] = data._invCov2D_opacities[gaussian_idx];
            s_means2D[i] = data._means2D[gaussian_idx];
            s_colors[i] = data._colors[gaussian_idx];
        }
        __syncthreads();

        #pragma unroll
        for(uint32_t i = 0; i < batchSize && T >= T_THRESHOLD; ++i) {
            evaluateGaussianOptimized(s_invCov2D_opacity[i], s_means2D[i], s_colors[i], outColor, x, y, T);
        }
    }

    d_img_out[pix_idx] = outColor;
}

void Render_GPU::run(DatasetGPU &data, float3 *d_img_out)
{
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks((data._dims.x + threadsPerBlock.x - 1) / threadsPerBlock.x, (data._dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    render<<<numBlocks, threadsPerBlock>>>(data, d_img_out);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
