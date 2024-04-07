
#include "dataset.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <numeric>

#include <json/json.hpp>
using json = nlohmann::json;

#include "helper/cuda_helper_host.h"

template <typename T>
bool readFile(std::vector<T> &vec, std::filesystem::path data_dir, const char filename[])
{
    std::filesystem::path input_file = data_dir / std::filesystem::path(filename);
    std::ifstream stream(input_file.c_str(), std::ios::in | std::ios::binary);

    if (!stream)
        return false;

    stream.seekg(0, std::ios::end);
    size_t filesize = stream.tellg();
    stream.seekg(0, std::ios::beg);

    vec.resize(filesize / sizeof(T));
    stream.read((char *)vec.data(), filesize);

    return true;
}

bool Dataset::load(fs::path input_dir)
{
    std::ifstream frameinfo_file(input_dir / "frameinfo.json");
    if (!frameinfo_file.is_open())
    {
        std::cout << "Could not load frameinfo file!" << std::endl;
        return false;
    }
    json frameinfo_json_data = json::parse(frameinfo_file);

    _dims.x = frameinfo_json_data["width"].get<uint32_t>();
    _dims.y = frameinfo_json_data["height"].get<uint32_t>();
    _n_pixels = _dims.x * _dims.y;

    _tile_dims = make_uint2(ceil(_dims.x / (float) TILE_SIZE), ceil(_dims.y / (float) TILE_SIZE));
    _n_tiles = _tile_dims.x * _tile_dims.y;

    std::cout << "Loading data... " << std::endl;
    readFile(_tile_ranges, input_dir, "ranges.dat");
    assert(_tile_ranges.size() >= _n_tiles && "Expecting one range per tile");

     // necessary because of a problem with provided data ranges
    _tile_ranges = { _tile_ranges.begin(), _tile_ranges.begin() + _n_tiles };
    uint32_t prev_end = _tile_ranges[0].y;
    for (uint32_t i = 1; i < _n_tiles; i++)
    {
        if (_tile_ranges[i].x != prev_end)
            _tile_ranges[i] = {prev_end, prev_end};

        prev_end = _tile_ranges[i].y;
    }

    _n_splat_entries = _tile_ranges[_tile_ranges.size() - 1].y;

    readFile(_tile_splat_entries, input_dir, "point_list.dat");
    assert(_tile_splat_entries.size() == _n_splat_entries && "Wrong number of splat entries");

    readFile(_means2D, input_dir, "means2D.dat");
    readFile(_invCov2D_opacities, input_dir, "conic_opacity.dat");
    readFile(_colors, input_dir, "colors.dat");
    _n_gaussians = _means2D.size();
    assert(_colors.size() == _n_gaussians && _invCov2D_opacities.size() == _n_gaussians && "Inconsistent number of gaussians");
    
    return true;
}

DatasetGPU Dataset::upload()
{
    DatasetGPU data_gpu;

    data_gpu._dims = _dims;
    data_gpu._n_pixels = _n_pixels;

    data_gpu._tile_dims = _tile_dims;
    data_gpu._n_tiles = _n_tiles;

    data_gpu._tile_ranges = uploadVector(_tile_ranges);

    data_gpu._n_splat_entries = _n_splat_entries; 
    data_gpu._tile_splat_entries = uploadVector(_tile_splat_entries);

    data_gpu._n_gaussians = _n_gaussians;
    data_gpu._invCov2D_opacities = uploadVector(_invCov2D_opacities);
    data_gpu._means2D = uploadVector(_means2D);
    data_gpu._colors = uploadVector(_colors);

    return data_gpu;
}