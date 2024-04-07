
#include "render.h"
#include "helper/helper_math.h"

#include <cassert>
#include <iostream>
#include <numeric>
#include <functional>

void Render_CPU::run(Dataset &data, std::vector<float3> &img_out)
{
    for (uint32_t x = 0; x < data._dims.x; x++)
    {
        for (uint32_t y = 0; y < data._dims.y; y++)
        {
            const uint32_t pix_idx = x + y * data._dims.x;

            const uint32_t tile_idx = (x / TILE_SIZE) + (y / TILE_SIZE) * data._tile_dims.x;
            assert(tile_idx < data._n_tiles);

            const uint2 tile_range = data._tile_ranges[tile_idx];
            const uint32_t n_tile_entries = tile_range.y - tile_range.x;

            float T = 1.0f;
            float3 out_color = make_float3(0.0f);
            for (uint32_t j = tile_range.x; j < tile_range.y; j++)
            {
                // Iterate over all gaussians that were splat onto this tile
                const uint32_t gaussian_idx = data._tile_splat_entries[j];
                const float4 invCov2D_opacity = data._invCov2D_opacities[gaussian_idx];

                const float3 inv_cov2D = make_float3(invCov2D_opacity); // cov2D is symmetric, so 2x2 matrix only contains 3 unique entries
                const float opacity = invCov2D_opacity.w;

                const float2 mean2D = data._means2D[gaussian_idx];

                // Evaluate the 2D Gaussian at this pixel position
				const float2 d = { x - mean2D.x, y - mean2D.y }; // d = (x - mu)
				const float power = -0.5f * ((inv_cov2D.x * d.x * d.x + inv_cov2D.z * d.y * d.y) + 2.0f * inv_cov2D.y * d.x * d.y); // = -0.5 * (d^T x inv_cov x d)
				if (power > 0.0f)
					continue;

                // alpha = o * exp(-0.5 * ((x-mu)^T x inv_cov x (x-mu)))
				const float alpha = std::min(0.99f, opacity * std::exp(power));
				if (alpha < ALPHA_THRESHOLD)
					continue;
                    
                const float3 color = data._colors[gaussian_idx];

                // Blend the Gaussians front-to-back
                out_color += color * T * alpha;
                T *= (1.0f - alpha);

                if (T < T_THRESHOLD) // early stopping of full opacity reached (= no transmittance left)
                    break;
            }

            img_out[pix_idx] = out_color;
        }        
    }
}