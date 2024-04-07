
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image/stb_image.h>
#include <stb_image/stb_image_write.h>

#include <json/json.hpp>
using json = nlohmann::json;

#include "cuda_runtime.h"
#include "helper/cuda_helper_host.h"
#include "helper/CPUTimer.h"
#include "helper/GPUTimer.cuh"
#include "helper/helper_math.h"

#include "dataset.h"
#include "render.h"

static constexpr int benchmark_iterations_cpu{1};
static constexpr int benchmark_iterations_gpu{1000};

uint to_8bit(const float f)
{
    return std::min(255, std::max(0, int(f * 256.f)));
}
uchar4 color_to_uchar4(const float3 color)
{
    uchar4 data;
    data.x = to_8bit(color.x);
    data.y = to_8bit(color.y);
    data.z = to_8bit(color.z);
    data.w = 255U;
    return data;
}
uchar4 color_to_uchar4(const float color)
{
    return color_to_uchar4(make_float3(color));
}
float mse_to_psnr(float mse)
{
    return -10.0f * std::log(mse) / std::log(10.0f);
}
template <typename T>
void writeImgToFile(std::vector<T>& img_out_float, int width, int height, fs::path output_dir, const std::string filename, int channels = 4)
{
    std::vector<uchar4> image(width * height);

    for (int i = 0; i < width * height; i++)
    {
        image[i] = color_to_uchar4(img_out_float[i]);
    }

    std::string output_file = (output_dir / filename).string();
    stbi_write_png(output_file.c_str(), width, height, channels, image.data(), width * channels);
}

int main(int argc, char* argv[])
{
	std::cout << "Assignment 03 - 3DGS Rendering" << std::endl;

	if(argc != 2)
	{
		std::cout << "Usage: ./3dgs-rendering <input_folder>" << std::endl;
		return -1;
	}

    fs::path input_dir = argv[1];

    Dataset data;
    if (!data.load(input_dir))
        return -1;

    std::cout << "Data loading done!" << std::endl << std::endl;

    uint32_t width = data._dims.x;
    uint32_t height = data._dims.y;

    std::vector<float3> img_out_cpu(width * height);

    constexpr bool run_cpu = benchmark_iterations_cpu > 0;
    if (run_cpu)
    {
        Render_CPU render_cpu;
        CPUTimer timer_cpu;

        std::cout << "Running CPU for " << benchmark_iterations_cpu << " iteration(s) ..." << std::endl;
        timer_cpu.start();
        for (int i = 0; i < benchmark_iterations_cpu; i++)
        {
            render_cpu.run(data, img_out_cpu);
        }
        float cpu_time = timer_cpu.end() / benchmark_iterations_cpu;

	    printf("CPU: %.6f ms/it\n\n", cpu_time);
    }
    else
    {
        std::cout << "Not running CPU!" << std::endl;
    }

    std::cout << "Uploading dataset to GPU ..." << std::endl;
    DatasetGPU data_gpu = data.upload();

    float3* d_img_out_gpu;
    CUDA_CHECK_THROW(cudaMalloc(&d_img_out_gpu, sizeof(float3) * width * height));

    std::cout << "Running GPU for " << benchmark_iterations_gpu << " iterations ..." << std::endl;
    Render_GPU render_gpu;
    render_gpu.setup(data._dims, data._tile_dims, data._n_gaussians);

    GPUTimer timer_gpu;
    for (int i = 0; i < benchmark_iterations_gpu; i++)
    {
        timer_gpu.start();
        render_gpu.run(data_gpu, d_img_out_gpu);
        (void) timer_gpu.end();
        //cudaDeviceSynchronize();
    }
    float gpu_time = timer_gpu.mean();

    render_gpu.finalize();
    printf("GPU: %.6f ms/it\n\n", gpu_time);

    std::vector<float> diff_img_mse(width * height);
    std::vector<float3> img_out_gpu(width * height);
    CUDA_CHECK_THROW(cudaMemcpy(img_out_gpu.data(), d_img_out_gpu, sizeof(float3) * width * height, cudaMemcpyDeviceToHost));

    float mse = 0.0f;
    bool success = true;
    if (run_cpu)
    {
        auto mse_fun = [](float3 left, float3 right) 
        { 
            float3 diff = left - right; 
            return dot(diff, diff) / 3.0f;
        };
        std::transform(img_out_cpu.begin(), img_out_cpu.end(), img_out_gpu.begin(), diff_img_mse.begin(), mse_fun);
        mse = std::reduce(diff_img_mse.begin(), diff_img_mse.end()) / float(width * height);
        success = mse < 1e-4f;

	    printf("Diff between CPU/GPU: MSE = %.6f, PSNR = %.2f (%s)\n", mse, mse_to_psnr(mse), success ? "SUCCESS" : "FAILED");
    }


    std::cout << "Writing images to file ..." << std::endl;
    writeImgToFile(img_out_gpu, width, height, input_dir, "out_gpu.png");
    if (run_cpu)
    {
        writeImgToFile(img_out_cpu, width, height, input_dir, "out_cpu.png");
        writeImgToFile(diff_img_mse, width, height, input_dir, "diff.png");
    }


    std::cout << "Writing results.csv file ..." << std::endl;
    std::ofstream results_csv;
    results_csv.open("results.csv", std::ios_base::app);
    results_csv << input_dir.filename().c_str() << "," << gpu_time << "," << mse << "," << (success ? "1" : "0") << std::endl;
    results_csv.close();

    return 0;
}