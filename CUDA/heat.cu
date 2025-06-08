#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>

#define IDX(i, j, ny) ((i) * (ny) + (j))

const float ALPHA = 0.01f;
const float DX = 0.02f;
const float DY = 0.02f;
const float DT = 0.0005f;


void get_color(float value, uint8_t &r, uint8_t &g, uint8_t &b) {
    if (value >= 500.0f)        { r = 255; g = 0;   b = 0;   }
    else if (value >= 100.0f)   { r = 255; g = 128; b = 0;   }
    else if (value >= 50.0f)    { r = 171; g = 71;  b = 188; }
    else if (value >= 25.0f)    { r = 255; g = 255; b = 0;   }
    else if (value >= 1.0f)     { r = 0;   g = 0;   b = 255; }
    else if (value >= 0.1f)     { r = 5;   g = 248; b = 252; }
    else                        { r = 255; g = 255; b = 255; }
}

void write_bmp(const char* filename, float* grid, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file %s for writing.\n", filename);
        return;
    }

    const int padding = (4 - (width * 3) % 4) % 4;
    const int file_size = 54 + (3 * width + padding) * height;

    uint8_t header[54] = {
        'B', 'M',               // Signature
        0,0,0,0,                // File size in bytes
        0,0, 0,0,               // Reserved
        54,0,0,0,               // Offset to image data
        40,0,0,0,               // Info header size
        0,0,0,0,                // Width
        0,0,0,0,                // Height
        1,0,                    // Planes
        24,0,                   // Bits per pixel
        0,0,0,0,                // Compression (0 = none)
        0,0,0,0,                // Image size (can be 0 for BI_RGB)
        0,0,0,0, 0,0,0,0,       // X/Y resolution
        0,0,0,0, 0,0,0,0        // Color palette
    };

    // Set width and height
    header[18] =  width        & 0xFF;
    header[19] = (width  >> 8) & 0xFF;
    header[20] = (width  >>16) & 0xFF;
    header[21] = (width  >>24) & 0xFF;
    header[22] =  height        & 0xFF;
    header[23] = (height >> 8) & 0xFF;
    header[24] = (height >>16) & 0xFF;
    header[25] = (height >>24) & 0xFF;

    // Set file size
    header[2] =  file_size        & 0xFF;
    header[3] = (file_size  >> 8) & 0xFF;
    header[4] = (file_size  >>16) & 0xFF;
    header[5] = (file_size  >>24) & 0xFF;

    fwrite(header, 1, 54, f);

    // Write pixel data (bottom to top)
    for (int i = height - 1; i >= 0; --i) {
        for (int j = 0; j < width; ++j) {
            uint8_t r, g, b;
            get_color(grid[i * width + j], r, g, b);
            fwrite(&b, 1, 1, f);
            fwrite(&g, 1, 1, f);
            fwrite(&r, 1, 1, f);
        }
        for (int k = 0; k < padding; ++k)
            fputc(0, f);
    }

    fclose(f);
    printf("BMP image written to %s\n", filename);
}

__global__
void heat_step_kernel(float* current, float* next, int nx, int ny, float r) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        next[IDX(i,j,ny)] = current[IDX(i,j,ny)]
            + r * (current[IDX(i+1,j,ny)] + current[IDX(i-1,j,ny)] - 2.0f * current[IDX(i,j,ny)])
            + r * (current[IDX(i,j+1,ny)] + current[IDX(i,j-1,ny)] - 2.0f * current[IDX(i,j,ny)]);
    }
}

void initialize(float* grid, int nx, int ny, float temp_source) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            if (i == j || i == ny - j - 1) {
                grid[IDX(i,j,ny)] = temp_source;
            } else {
                grid[IDX(i,j,ny)] = 0.0f;
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: ./heat_cuda nx ny steps blockSize\n";
        return 1;
    }

    int nx, ny = atoi(argv[1]);
    int steps = atoi(argv[2]);
    int blockSize = atoi(argv[3]);

    float r = ALPHA * DT / (DX * DY);
    size_t size = nx * ny * sizeof(float);

    float* h_grid = (float*)malloc(size);
    float* h_result = (float*)malloc(size);
    initialize(h_grid, nx, ny, 1500.0f);

    float *d_current, *d_next;
    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);
    cudaMemcpy(d_current, h_grid, size, cudaMemcpyHostToDevice);

    dim3 block(blockSize, blockSize);
    dim3 grid((ny + block.x - 1) / block.x, (nx + block.y - 1) / block.y);

    for (int t = 0; t < steps; t++) {
        heat_step_kernel<<<grid, block>>>(d_current, d_next, nx, ny, r);
        std::swap(d_current, d_next);
    }

    cudaMemcpy(h_result, d_current, size, cudaMemcpyDeviceToHost);

    write_bmp("output.bmp", h_result, nx, ny);

    cudaFree(d_current);
    cudaFree(d_next);
    free(h_grid);
    free(h_result);

    return 0;
}
