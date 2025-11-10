// mandelbrot_cuda.cu
// Compilar: nvcc -O3 mandelbrot_cuda.cu -o mandelbrot
// Executar: ./mandelbrot mandelbrot.ppm

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA error %s:%d: %s\n",             \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// Mapeamento simples de 'iter' para cor RGB
__device__ void iteration_to_color(int iter, int maxIter, unsigned char &r, unsigned char &g, unsigned char &b) {
    if (iter >= maxIter) {
        r = g = b = 0;
        return;
    }
    float t = (float)iter / (float)maxIter;
    r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
    g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
    b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
}

// Kernel: cada thread calcula 1 pixel
__global__ void mandelbrot_kernel(unsigned char* img, int width, int height,
                                  float xmin, float xmax, float ymin, float ymax,
                                  int maxIter)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int idx = (py * width + px) * 3; // 3 canais RGB

    // Map pixel -> ponto complexo c
    float x0 = xmin + (xmax - xmin) * (px + 0.5f) / (float)width;
    float y0 = ymin + (ymax - ymin) * (py + 0.5f) / (float)height;

    float x = 0.0f;
    float y = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    int iter = 0;

    while (x2 + y2 <= 4.0f && iter < maxIter) {
        y = 2.0f * x * y + y0;
        x = x2 - y2 + x0;
        x2 = x * x;
        y2 = y * y;
        iter++;
    }

    unsigned char r,g,b;
    iteration_to_color(iter, maxIter, r, g, b);

    img[idx + 0] = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

// escreve PPM (P6)
void write_ppm(const char* filename, unsigned char* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        perror("fopen");
        exit(EXIT_FAILURE);
    }
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    size_t total = (size_t)width * (size_t)height * 3;
    fwrite(data, 1, total, f);
    fclose(f);
}

double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char** argv) {
    const char* outname = "mandelbrot.ppm";
    if (argc >= 2) outname = argv[1];

    // parâmetros (muda se quiser)
    const int width = 1280;
    const int height = 720;
    const int maxIter = 1000;

    const float xmin = -2.0f;
    const float xmax = 1.0f;
    const float ymin = -1.0f;
    const float ymax = 1.0f;

    size_t img_size = (size_t)width * (size_t)height * 3; // bytes

    unsigned char* h_img = (unsigned char*)malloc(img_size);
    if (!h_img) {
        fprintf(stderr, "Falha ao alocar memória host\n");
        return 1;
    }

    unsigned char* d_img = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_img, img_size));

    // blocos e grade
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    double t0 = now_seconds();
    mandelbrot_kernel<<<grid, block>>>(d_img, width, height, xmin, xmax, ymin, ymax, maxIter);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double t1 = now_seconds();

    CUDA_CHECK(cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost));

    write_ppm(outname, h_img, width, height);

    printf("Imagem salva em '%s' (%dx%d), maxIter=%d\n", outname, width, height, maxIter);
    printf("Tempo kernel+sync: %.4f s\n", t1 - t0);

    CUDA_CHECK(cudaFree(d_img));
    free(h_img);
    return 0;
}
