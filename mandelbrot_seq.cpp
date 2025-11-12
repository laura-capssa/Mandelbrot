#include <iostream>
#include <fstream>
#include <chrono>

int main() {
    int width = 1920, height = 1080;
    int maxIter = 500;
    float zoom = 1.0f, offsetX = -0.5f, offsetY = 0.0f;

    unsigned char *image = new unsigned char[width * height * 3];

    auto start = std::chrono::high_resolution_clock::now();

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float jx = 1.5f * (x - width / 2) / (0.5f * zoom * width) + offsetX;
            float jy = (y - height / 2) / (0.5f * zoom * height) + offsetY;

            float zx = 0, zy = 0;
            int iter = 0;

            while (zx * zx + zy * zy < 4.0f && iter < maxIter) {
                float tmp = zx * zx - zy * zy + jx;
                zy = 2.0f * zx * zy + jy;
                zx = tmp;
                iter++;
            }

            int idx = (y * width + x) * 3;
            if (iter == maxIter) {
                image[idx] = image[idx+1] = image[idx+2] = 0;
            } else {
                float t = (float)iter / maxIter;
                image[idx]   = (unsigned char)(9 * (1 - t) * t * t * t * 255);
                image[idx+1] = (unsigned char)(15 * (1 - t) * (1 - t) * t * t * 255);
                image[idx+2] = (unsigned char)(8.5 * (1 - t) * (1 - t) * (1 - t) * t * 255);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(end - start).count();

    std::cout << "\nTempo total (sequencial): " << totalTime << " segundos\n";

    std::ofstream file("mandelbrot_seq.ppm");
    file << "P3\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height * 3; i += 3) {
        file << (int)image[i] << " " << (int)image[i+1] << " " << (int)image[i+2] << "\n";
    }
    file.close();

    delete[] image;
    return 0;
}
