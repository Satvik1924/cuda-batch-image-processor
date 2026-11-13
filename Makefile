NVCC = nvcc
NVCC_FLAGS = --std=c++17 -O3 -Isrc -Wno-deprecated-gpu-targets
TARGET = bin/image_processor
SRC = src/image_processor.cu

.PHONY: all build clean

all: build

build: $(TARGET)

$(TARGET): $(SRC) src/image_processor.h
	mkdir -p bin
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -rf bin input_data output_data artifacts proof_of_execution.tar.gz
