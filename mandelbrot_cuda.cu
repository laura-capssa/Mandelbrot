#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>      // Para medir tempo
#include <cuda_runtime.h>

// Verifica erros do CUDA (para facilitar debug)
#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "Erro CUDA %s:%d: %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while(0)

// Função auxiliar para medir tempo em segundos
double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Função para definir a cor de cada pixel
__device__ void iteration_to_color(int iter, int maxIter,
                                   unsigned char &r, unsigned char &g, unsigned char &b) {
    if (iter == maxIter) {
        // Pontos "dentro" do fractal — fundo preto
        r = g = b = 0;
    } else {
        // Gradiente de cores baseado em iterações
        float t = (float)iter / (float)maxIter;
        r = (unsigned char)(9 * (1 - t) * t * t * t * 255);
        g = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
        b = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
    }
}

// Cada thread calcula um pixel
__global__ void julia_kernel(unsigned char* img, int width, int height,
                             float xmin, float xmax, float ymin, float ymax,
                             int maxIter, float cx, float cy) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    int idx = (py * width + px) * 3; // posição RGB do pixel

    // Converter coordenadas de pixel → coordenadas complexas (x0, y0)
    float x = xmin + (xmax - xmin) * px / (float)width;
    float y = ymin + (ymax - ymin) * py / (float)height;

    int iter = 0;

    // Fórmula do conjunto de Julia:
    // z = z² + c  (c é fixo e muda o formato)
    while (x * x + y * y <= 4.0f && iter < maxIter) {
        float xtemp = x * x - y * y + cx;
        y = 2.0f * x * y + cy;
        x = xtemp;
        iter++;
    }

    unsigned char r, g, b;
    iteration_to_color(iter, maxIter, r, g, b);

    img[idx + 0] = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

// Salva imagem no formato PPM (simples)
void write_ppm(const char* filename, unsigned char* data, int width, int height) {
    FILE* f = fopen(filename, "wb");
    fprintf(f, "P6\n%d %d\n255\n", width, height);
    fwrite(data, 1, width * height * 3, f);
    fclose(f);
}

int main(int argc, char** argv) {
    const char* filename = (argc >= 2) ? argv[1] : "julia.ppm";

    const int width = 1280, height = 720;
    const int maxIter = 1000;

    // Área do plano complexo
    const float xmin = -1.5f, xmax = 1.5f;
    const float ymin = -1.0f, ymax = 1.0f;

    // Constante c do fractal (muda o formato! experimente valores diferentes!)
    const float cx = -0.7f;
    const float cy = 0.27015f;

    size_t img_size = width * height * 3;
    unsigned char* h_img = (unsigned char*)malloc(img_size);
    unsigned char* d_img = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_img, img_size));

    // Configuração de blocos e threads
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    // ⏱️ Marca o tempo antes do kernel
    double t0 = now_seconds();

    // Executa o kernel CUDA
    julia_kernel<<<grid, block>>>(d_img, width, height, xmin, xmax, ymin, ymax, maxIter, cx, cy);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ⏱️ Marca o tempo após o kernel
    double t1 = now_seconds();
    printf("Tempo total de execução do kernel: %.4f segundos\n", t1 - t0);

    CUDA_CHECK(cudaMemcpy(h_img, d_img, img_size, cudaMemcpyDeviceToHost));

    write_ppm(filename, h_img, width, height);
    printf("Imagem salva em '%s'\n", filename);

    free(h_img);
    CUDA_CHECK(cudaFree(d_img));
    return 0;
}

// ---- COMANDOS TERMINAL ----
// nvcc -O3 mandelbrot_cuda.cu -o mandelbrot
// ./mandelbrot mandelbrot.ppm
// convert mandelbrot.ppm mandelbrot.png
// xdg-open mandelbrot.png
