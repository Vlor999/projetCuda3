
#include "render.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"

#include <cub/cub.cuh>

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

__device__ void evaluateGaussian(const DatasetGPU &data, float3 &outColor, const uint32_t x, const uint32_t y, const uint32_t gaussian_idx, float &T)
{
    if (T < T_THRESHOLD) { // early stopping of full opacity reached (= no transmittance left)
        return;
    }

    const float4 invCov2D_opacity = data._invCov2D_opacities[gaussian_idx];
    const float3 inv_cov2D = make_float3(invCov2D_opacity);
    const float opacity = invCov2D_opacity.w;
    const float2 mean2D = data._means2D[gaussian_idx];

    const float dx = x - mean2D.x;
    const float dy = y - mean2D.y;
    const float power = -0.5f * (inv_cov2D.x * dx * dx + 2.0f * inv_cov2D.y * dx * dy + inv_cov2D.z * dy * dy);
    
    if (power > 0.0f){
        return;
    }

    float alpha = opacity * __expf(power);
    if (alpha < ALPHA_THRESHOLD) {
        return;
    }
    alpha = fminf(alpha, 0.99f);

    const float3 color = data._colors[gaussian_idx];
    outColor += color * T * alpha;
    T *= (1.0f - alpha);
}

__global__ void render(const DatasetGPU data, float3 *__restrict__ d_img_out)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= data._dims.x || y >= data._dims.y) return;

    const uint32_t pix_idx = x + y * data._dims.x;
    const uint32_t tile_x = x / TILE_SIZE;
    const uint32_t tile_y = y / TILE_SIZE;
    const uint32_t tile_idx = tile_x + tile_y * data._tile_dims.x;

    if (tile_idx >= data._n_tiles) return;
    const uint2 tile_range = data._tile_ranges[tile_idx];

    float3 outColor = make_float3(0.0f);
    float T = 1.0f;

    for (uint32_t i = tile_range.x; i <= tile_range.y; ++i) {
        const uint32_t gaussian_idx = data._tile_splat_entries[i];
        evaluateGaussian(data, outColor, x, y, gaussian_idx, T);
        if (T < T_THRESHOLD){
            break;
        }
    }

    d_img_out[pix_idx] = outColor;
}

void Render_GPU::run(DatasetGPU &data, float3 *d_img_out)
{
    const dim3 threadsPerBlock(16, 16);
    const dim3 numBlocks(
        (data._dims.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (data._dims.y + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    render<<<numBlocks, threadsPerBlock>>>(data, d_img_out);
}
