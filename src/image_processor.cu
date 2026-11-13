#include "image_processor.h"

#include <cuda_runtime.h>
#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace cuda_img_proc {

namespace {

inline void CheckCudaError(cudaError_t err, const char* context) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA Error (" << context
              << "): " << cudaGetErrorString(err) << "\n";
    std::exit(EXIT_FAILURE);
  }
}

__device__ inline int ClampIndex(int val, int min_val, int max_val) {
  return max(min_val, min(val, max_val));
}

// Loads a 16x16 block + 1-pixel halo into an 18x18 __shared__ memory tile.
__device__ void LoadTileWithHalo(const uint8_t* __restrict__ input,
                                 uint8_t tile[kSharedTileDim][kSharedTileDim],
                                 int width, int height) {
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int col = blockIdx.x * kBlockSize + tx;
  const int row = blockIdx.y * kBlockSize + ty;

  const int clamped_col = ClampIndex(col, 0, width - 1);
  const int clamped_row = ClampIndex(row, 0, height - 1);
  tile[ty + kHaloRadius][tx + kHaloRadius] =
      input[clamped_row * width + clamped_col];

  // Left and right halos
  if (tx == 0) {
    const int left_col = ClampIndex(col - 1, 0, width - 1);
    tile[ty + kHaloRadius][0] = input[clamped_row * width + left_col];
  }
  if (tx == kBlockSize - 1) {
    const int right_col = ClampIndex(col + 1, 0, width - 1);
    tile[ty + kHaloRadius][kSharedTileDim - 1] =
        input[clamped_row * width + right_col];
  }

  // Top and bottom halos
  if (ty == 0) {
    const int top_row = ClampIndex(row - 1, 0, height - 1);
    tile[0][tx + kHaloRadius] = input[top_row * width + clamped_col];
  }
  if (ty == kBlockSize - 1) {
    const int bot_row = ClampIndex(row + 1, 0, height - 1);
    tile[kSharedTileDim - 1][tx + kHaloRadius] =
        input[bot_row * width + clamped_col];
  }

  // Four corner halos
  if (tx == 0 && ty == 0) {
    tile[0][0] = input[ClampIndex(row - 1, 0, height - 1) * width +
                       ClampIndex(col - 1, 0, width - 1)];
  }
  if (tx == kBlockSize - 1 && ty == 0) {
    tile[0][kSharedTileDim - 1] =
        input[ClampIndex(row - 1, 0, height - 1) * width +
              ClampIndex(col + 1, 0, width - 1)];
  }
  if (tx == 0 && ty == kBlockSize - 1) {
    tile[kSharedTileDim - 1][0] =
        input[ClampIndex(row + 1, 0, height - 1) * width +
              ClampIndex(col - 1, 0, width - 1)];
  }
  if (tx == kBlockSize - 1 && ty == kBlockSize - 1) {
    tile[kSharedTileDim - 1][kSharedTileDim - 1] =
        input[ClampIndex(row + 1, 0, height - 1) * width +
              ClampIndex(col + 1, 0, width - 1)];
  }
}

// Kernel 1: 3x3 Gaussian smoothing using 2D shared memory tiling.
__global__ void GaussianDenoiseSharedKernel(const uint8_t* __restrict__ input,
                                            uint8_t* __restrict__ output,
                                            int width, int height) {
  __shared__ uint8_t tile[kSharedTileDim][kSharedTileDim];
  LoadTileWithHalo(input, tile, width, height);
  __syncthreads();

  const int col = blockIdx.x * kBlockSize + threadIdx.x;
  const int row = blockIdx.y * kBlockSize + threadIdx.y;
  if (col >= width || row >= height) return;

  const int sy = threadIdx.y + kHaloRadius;
  const int sx = threadIdx.x + kHaloRadius;

  const int sum =
      1 * tile[sy - 1][sx - 1] + 2 * tile[sy - 1][sx] + 1 * tile[sy - 1][sx + 1] +
      2 * tile[sy][sx - 1]     + 4 * tile[sy][sx]     + 2 * tile[sy][sx + 1] +
      1 * tile[sy + 1][sx - 1] + 2 * tile[sy + 1][sx] + 1 * tile[sy + 1][sx + 1];

  output[row * width + col] = static_cast<uint8_t>(sum >> 4);
}

// Kernel 2: 3x3 Sobel edge detection using 2D shared memory tiling.
__global__ void SobelEdgeDetectSharedKernel(const uint8_t* __restrict__ input,
                                            uint8_t* __restrict__ output,
                                            int width, int height) {
  __shared__ uint8_t tile[kSharedTileDim][kSharedTileDim];
  LoadTileWithHalo(input, tile, width, height);
  __syncthreads();

  const int col = blockIdx.x * kBlockSize + threadIdx.x;
  const int row = blockIdx.y * kBlockSize + threadIdx.y;
  if (col >= width || row >= height) return;

  const int sy = threadIdx.y + kHaloRadius;
  const int sx = threadIdx.x + kHaloRadius;

  const int gx =
      -1 * tile[sy - 1][sx - 1] + 1 * tile[sy - 1][sx + 1] +
      -2 * tile[sy][sx - 1]     + 2 * tile[sy][sx + 1] +
      -1 * tile[sy + 1][sx - 1] + 1 * tile[sy + 1][sx + 1];

  const int gy =
      -1 * tile[sy - 1][sx - 1] - 2 * tile[sy - 1][sx] - 1 * tile[sy - 1][sx + 1] +
       1 * tile[sy + 1][sx - 1] + 2 * tile[sy + 1][sx] + 1 * tile[sy + 1][sx + 1];

  const int magnitude = static_cast<int>(
      sqrtf(static_cast<float>(gx * gx + gy * gy)));
  output[row * width + col] =
      static_cast<uint8_t>(min(255, max(0, magnitude)));
}

void PrintUsage(const char* prog_name) {
  std::cout << "Usage: " << prog_name << " [options]\n"
            << "Options:\n"
            << "  --input_dir <path>   Directory containing .pgm images (default: input_data)\n"
            << "  --output_dir <path>  Directory for processed .pgm images (default: output_data)\n"
            << "  --filter <mode>      Filter mode: gaussian | sobel | pipeline (default: pipeline)\n"
            << "  --max_images <N>     Max images to process, 0 = all (default: 0)\n"
            << "  --help               Show this help message\n";
}

bool HasPgmExtension(const std::string& filename) {
  if (filename.size() < 4) return false;
  return filename.substr(filename.size() - 4) == ".pgm";
}

std::vector<std::string> ListPgmFiles(const std::string& dir_path) {
  std::vector<std::string> files;
  DIR* dir = opendir(dir_path.c_str());
  if (dir == nullptr) {
    std::cerr << "Error: Cannot open directory '" << dir_path << "'\n";
    return files;
  }
  struct dirent* entry = nullptr;
  while ((entry = readdir(dir)) != nullptr) {
    std::string name(entry->d_name);
    if (HasPgmExtension(name)) {
      files.push_back(name);
    }
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

void SkipPgmCommentsAndWhitespace(std::istream* stream) {
  while (true) {
    int ch = stream->peek();
    if (ch == EOF) break;
    if (std::isspace(ch)) {
      stream->get();
    } else if (ch == '#') {
      std::string comment;
      std::getline(*stream, comment);
    } else {
      break;
    }
  }
}

}  // namespace

bool ParseArguments(int argc, char** argv, ProgramOptions* options) {
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return false;
    } else if (arg == "--input_dir" && i + 1 < argc) {
      options->input_dir = argv[++i];
    } else if (arg == "--output_dir" && i + 1 < argc) {
      options->output_dir = argv[++i];
    } else if (arg == "--filter" && i + 1 < argc) {
      std::string mode_str(argv[++i]);
      if (mode_str == "gaussian") {
        options->mode = FilterMode::kGaussian;
      } else if (mode_str == "sobel") {
        options->mode = FilterMode::kSobel;
      } else if (mode_str == "pipeline") {
        options->mode = FilterMode::kPipeline;
      } else {
        std::cerr << "Invalid filter mode: " << mode_str << "\n";
        return false;
      }
    } else if (arg == "--max_images" && i + 1 < argc) {
      options->max_images = std::atoi(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n";
      PrintUsage(argv[0]);
      return false;
    }
  }
  return true;
}

bool LoadPgmImage(const std::string& file_path, PgmImage* image) {
  std::ifstream file(file_path, std::ios::binary);
  if (!file.is_open()) return false;

  std::string magic;
  file >> magic;
  if (magic != "P5") return false;

  SkipPgmCommentsAndWhitespace(&file);
  file >> image->width;
  SkipPgmCommentsAndWhitespace(&file);
  file >> image->height;
  SkipPgmCommentsAndWhitespace(&file);
  int max_val = 0;
  file >> max_val;
  file.get();  // Consume single whitespace byte before binary payload

  if (image->width <= 0 || image->height <= 0 || max_val <= 0) return false;

  const size_t num_bytes =
      static_cast<size_t>(image->width) * static_cast<size_t>(image->height);
  image->pixels.resize(num_bytes);
  file.read(reinterpret_cast<char*>(image->pixels.data()), num_bytes);
  return file.gcount() == static_cast<std::streamsize>(num_bytes);
}

bool SavePgmImage(const std::string& file_path, const PgmImage& image) {
  std::ofstream file(file_path, std::ios::binary);
  if (!file.is_open()) return false;

  file << "P5\n" << image.width << " " << image.height << "\n255\n";
  file.write(reinterpret_cast<const char*>(image.pixels.data()),
             image.pixels.size());
  return file.good();
}

BatchMetrics ProcessImageBatch(const ProgramOptions& options) {
  BatchMetrics metrics;
  mkdir(options.output_dir.c_str(), 0755);

  std::vector<std::string> files = ListPgmFiles(options.input_dir);
  if (files.empty()) {
    std::cerr << "No .pgm images found in " << options.input_dir << "\n";
    return metrics;
  }

  if (options.max_images > 0 &&
      static_cast<size_t>(options.max_images) < files.size()) {
    files.resize(options.max_images);
  }

  cudaDeviceProp prop;
  CheckCudaError(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
  std::cout << "=== CUDA Batch Image Processor ===\n"
            << "GPU Device       : " << prop.name << " ("
            << prop.multiProcessorCount << " SMs)\n"
            << "Input Directory  : " << options.input_dir << "\n"
            << "Output Directory : " << options.output_dir << "\n"
            << "Batch Size       : " << files.size() << " images\n\n";

  uint8_t* d_in = nullptr;
  uint8_t* d_tmp = nullptr;
  uint8_t* d_out = nullptr;
  size_t allocated_bytes = 0;

  cudaEvent_t start_evt, stop_evt;
  CheckCudaError(cudaEventCreate(&start_evt), "cudaEventCreate start");
  CheckCudaError(cudaEventCreate(&stop_evt), "cudaEventCreate stop");

  for (size_t idx = 0; idx < files.size(); ++idx) {
    const std::string in_path = options.input_dir + "/" + files[idx];
    const std::string out_path = options.output_dir + "/" + files[idx];

    PgmImage img;
    if (!LoadPgmImage(in_path, &img)) {
      std::cerr << "Warning: Skipping unreadable image " << in_path << "\n";
      continue;
    }

    const size_t num_bytes = img.pixels.size();
    if (num_bytes > allocated_bytes) {
      if (d_in != nullptr) cudaFree(d_in);
      if (d_tmp != nullptr) cudaFree(d_tmp);
      if (d_out != nullptr) cudaFree(d_out);
      CheckCudaError(cudaMalloc(&d_in, num_bytes), "cudaMalloc d_in");
      CheckCudaError(cudaMalloc(&d_tmp, num_bytes), "cudaMalloc d_tmp");
      CheckCudaError(cudaMalloc(&d_out, num_bytes), "cudaMalloc d_out");
      allocated_bytes = num_bytes;
    }

    // Host-to-Device transfer
    cudaEventRecord(start_evt);
    CheckCudaError(
        cudaMemcpy(d_in, img.pixels.data(), num_bytes, cudaMemcpyHostToDevice),
        "cudaMemcpy H2D");
    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float h2d_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, start_evt, stop_evt);

    // Kernel execution
    dim3 block(kBlockSize, kBlockSize);
    dim3 grid((img.width + kBlockSize - 1) / kBlockSize,
              (img.height + kBlockSize - 1) / kBlockSize);

    cudaEventRecord(start_evt);
    if (options.mode == FilterMode::kGaussian) {
      GaussianDenoiseSharedKernel<<<grid, block>>>(d_in, d_out, img.width,
                                                   img.height);
    } else if (options.mode == FilterMode::kSobel) {
      SobelEdgeDetectSharedKernel<<<grid, block>>>(d_in, d_out, img.width,
                                                   img.height);
    } else {
      GaussianDenoiseSharedKernel<<<grid, block>>>(d_in, d_tmp, img.width,
                                                   img.height);
      SobelEdgeDetectSharedKernel<<<grid, block>>>(d_tmp, d_out, img.width,
                                                   img.height);
    }
    CheckCudaError(cudaGetLastError(), "Kernel launch");
    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start_evt, stop_evt);

    // Device-to-Host transfer
    PgmImage out_img;
    out_img.width = img.width;
    out_img.height = img.height;
    out_img.pixels.resize(num_bytes);

    cudaEventRecord(start_evt);
    CheckCudaError(cudaMemcpy(out_img.pixels.data(), d_out, num_bytes,
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float d2h_ms = 0.0f;
    cudaEventElapsedTime(&d2h_ms, start_evt, stop_evt);

    SavePgmImage(out_path, out_img);

    metrics.images_processed++;
    metrics.total_pixels += static_cast<long long>(img.width) * img.height;
    metrics.total_h2d_ms += h2d_ms;
    metrics.total_kernel_ms += kernel_ms;
    metrics.total_d2h_ms += d2h_ms;

    if ((idx + 1) % 25 == 0 || idx + 1 == files.size()) {
      std::cout << "[" << std::setw(3) << (idx + 1) << "/" << files.size()
                << "] Processed " << files[idx] << " (" << img.width << "x"
                << img.height << ") | Kernel: " << std::fixed
                << std::setprecision(4) << kernel_ms << " ms\n";
    }
  }

  cudaEventDestroy(start_evt);
  cudaEventDestroy(stop_evt);
  if (d_in != nullptr) cudaFree(d_in);
  if (d_tmp != nullptr) cudaFree(d_tmp);
  if (d_out != nullptr) cudaFree(d_out);

  return metrics;
}

}  // namespace cuda_img_proc

int main(int argc, char** argv) {
  cuda_img_proc::ProgramOptions options;
  if (!cuda_img_proc::ParseArguments(argc, argv, &options)) {
    return EXIT_FAILURE;
  }

  cuda_img_proc::BatchMetrics metrics =
      cuda_img_proc::ProcessImageBatch(options);

  const double megapixels = static_cast<double>(metrics.total_pixels) / 1e6;
  const double throughput_mpix_s =
      (metrics.total_kernel_ms > 0.0f)
          ? (megapixels / (metrics.total_kernel_ms / 1000.0))
          : 0.0;

  std::cout << "\n=== Batch Execution Summary ===\n"
            << "Images Processed      : " << metrics.images_processed << "\n"
            << "Total Pixels          : " << metrics.total_pixels << " ("
            << std::fixed << std::setprecision(2) << megapixels << " MPix)\n"
            << "Total H2D Time        : " << std::setprecision(3)
            << metrics.total_h2d_ms << " ms\n"
            << "Total GPU Kernel Time : " << metrics.total_kernel_ms << " ms\n"
            << "Total D2H Time        : " << metrics.total_d2h_ms << " ms\n"
            << "GPU Kernel Throughput : " << std::setprecision(2)
            << throughput_mpix_s << " MPixels/sec\n";

  return EXIT_SUCCESS;
}
