#ifndef CUDA_BATCH_IMAGE_PROCESSOR_IMAGE_PROCESSOR_H_
#define CUDA_BATCH_IMAGE_PROCESSOR_IMAGE_PROCESSOR_H_

#include <cstdint>
#include <string>
#include <vector>

namespace cuda_img_proc {

constexpr int kBlockSize = 16;
constexpr int kHaloRadius = 1;
constexpr int kSharedTileDim = kBlockSize + 2 * kHaloRadius;

enum class FilterMode {
  kGaussian,
  kSobel,
  kPipeline  // Gaussian denoising followed by Sobel edge detection
};

struct ProgramOptions {
  std::string input_dir = "input_data";
  std::string output_dir = "output_data";
  FilterMode mode = FilterMode::kPipeline;
  int max_images = 0;  // 0 means process all images in input_dir
};

struct PgmImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> pixels;
};

struct BatchMetrics {
  int images_processed = 0;
  long long total_pixels = 0;
  float total_kernel_ms = 0.0f;
  float total_h2d_ms = 0.0f;
  float total_d2h_ms = 0.0f;
};

// Parses CLI flags (--input_dir, --output_dir, --filter, --max_images).
bool ParseArguments(int argc, char** argv, ProgramOptions* options);

// Loads an 8-bit binary PGM (P5) image from disk.
bool LoadPgmImage(const std::string& file_path, PgmImage* image);

// Saves an 8-bit binary PGM (P5) image to disk.
bool SavePgmImage(const std::string& file_path, const PgmImage& image);

// Executes GPU filtering across all PGM images found in options.input_dir.
BatchMetrics ProcessImageBatch(const ProgramOptions& options);

}  // namespace cuda_img_proc

#endif  // CUDA_BATCH_IMAGE_PROCESSOR_IMAGE_PROCESSOR_H_
