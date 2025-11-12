
## Para rodar versão sequenial 
g++ mandelbrot_seq.cpp -o mandelbrot_seq
./mandelbrot_seq

## Para rodar versão paralela 
nvcc -O3 mandelbrot_cuda.cu -o mandelbrot
./mandelbrot mandelbrot.ppm
convert mandelbrot.ppm mandelbrot.png
xdg-open mandelbrot.png
