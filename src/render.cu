
#include "render.h"
#include "helper/helper_math.h"
#include "helper/cuda_helper_host.h"


Render_GPU::Render_GPU()
{
}

Render_GPU::~Render_GPU()
{
}

void Render_GPU::setup(const uint2 dims, const uint2 tile_dims, const uint32_t n_gaussians)
{
    uint32_t n_pixels = dims.x * dims.y;
    uint32_t n_tiles = tile_dims.x * tile_dims.y;
}

void Render_GPU::finalize()
{
}

void Render_GPU::run(DatasetGPU &data, float3 *d_img_out)
{
    /*
        TODO: Compute the color for each pixel of the output image (see Equations (2) and (3) in assignment sheet)
              Do not store and reuse calculations between different runs! You have to do the full computation each run!
    */
    return;
}