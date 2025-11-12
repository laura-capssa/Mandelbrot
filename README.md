
## Pré-requisitos

- Compilador C++ (para versão sequencial)
- NVIDIA CUDA Toolkit (para versão paralela)
- ImageMagick (para conversão de PPM para PNG)
- Um visualizador de imagens (xdg-open funciona no Linux)

## Como Compilar e Executar

### Versão Sequencial

Para compilar e executar a versão sequencial:

```bash
g++ mandelbrot_seq.cpp -o mandelbrot_seq
./mandelbrot_seq
```

### Versão Paralela

Para compilar e executar a versão paralela:

```bash
nvcc -O3 mandelbrot_cuda.cu -o mandelbrot
./mandelbrot mandelbrot.ppm
convert mandelbrot.ppm mandelbrot.png
xdg-open mandelbrot.png
```

Isso gerará uma imagem PPM, converterá para PNG e abrirá no visualizador padrão.
