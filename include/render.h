
#pragma once

#include "dataset.h"

constexpr float T_THRESHOLD = 1e-4f;
constexpr float ALPHA_THRESHOLD = 1.0f / 255.f;

struct Render_CPU
{
    Render_CPU() {};

    void run(Dataset& data, std::vector<float3>& img_out);
};

struct Render_GPU
{
    Render_GPU();
    ~Render_GPU();

    void setup(const uint2 dims, const uint2 tile_dims, const uint32_t n_gaussians);
    void finalize();

    void run(DatasetGPU& data, float3* d_img_out);
};