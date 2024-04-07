
#pragma once

#include <filesystem>
#include <vector>
namespace fs = std::filesystem;

#include "cuda_runtime.h"

constexpr uint32_t TILE_SIZE = 16;

struct DatasetGPU
{
    uint2 _dims; // (width (W), height (H))
    uint32_t _n_pixels; // N_P = W * H

    uint2 _tile_dims; // (TW = n tiles width (W/TILE_SIZE), TH = n tiles height (=H/TILE_SIZE))
    uint32_t _n_tiles; // N_T = TW * TH

    uint2* _tile_ranges;  // (from, to) range of tiles inside splat entries | size: N_T * 2 * sizeof(uint32_t)

    uint32_t _n_splat_entries; // N_S = _tile_ranges[N_T-1].y
    uint32_t* _tile_splat_entries; // indices of Gaussians per tile | size: N_S * 

    uint32_t _n_gaussians;  // N_G
    float4* _invCov2D_opacities; // 2D splat inverse covariance matrix + opacity | size: N_G * 4 * sizeof(float)
    float2* _means2D;            // 2D splat centers | size: N_G * 2 * sizeof(float)
    float3* _colors;             // view-dependent RGB color | size: N_G * 3 * sizeof(float)
};

struct Dataset
{
    uint2 _dims; // (width (W), height (H))
    uint32_t _n_pixels; // N_P = W * H

    uint2 _tile_dims; // (TW = n tiles width (W/TILE_SIZE), TH = n tiles height (=H/TILE_SIZE))
    uint32_t _n_tiles; // N_T = TW * TH
    std::vector<uint2> _tile_ranges;

    uint32_t _n_splat_entries; // N_S = sum(counts)
    std::vector<uint32_t> _tile_splat_entries; // indices of Gaussians per tile

    uint32_t _n_gaussians; // N_G
    std::vector<float4> _invCov2D_opacities;
    std::vector<float2> _means2D;
    std::vector<float3> _colors; // view-dependent RGB color

    bool load(fs::path input_dir);
    DatasetGPU upload();
};