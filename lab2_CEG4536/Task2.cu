#include <cuda_runtime.h>
#include <iostream>

__global__ void simplifiedReductionKernel(int* input, int* output, int size) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x * 2 + tid;

    // Charger les éléments d'entrée dans la mémoire partagée avec vérification des limites
    sharedData[tid] = (globalIndex < size) ? input[globalIndex] : 0;
    if (globalIndex + blockDim.x < size) {
        sharedData[tid] += input[globalIndex + blockDim.x];
    }
    __syncthreads();

    // Effectuer la réduction dans la mémoire partagée
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Écrire le résultat de ce bloc dans le tableau de sortie
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    const int size = 1024; // Nombre total d'éléments
    const int bytes = size * sizeof(int);
    const int blockSize = 256; // Nombre de threads par bloc
    const int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2); // Nombre de blocs

    int* h_input = new int[size];
    int* h_output = new int[gridSize];
    int* d_input, * d_output;

    // Initialiser le tableau d'entrée avec des 1 (ou toute autre valeur selon les instructions du labo)
    for (int i = 0; i < size; i++) {
        h_input[i] = 1;
    }

    // Allouer la mémoire sur le périphérique
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, gridSize * sizeof(int));
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Lancer le noyau de réduction avec une mémoire partagée dynamique
    simplifiedReductionKernel << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size);
    cudaDeviceSynchronize();

    // Copier les résultats partiels du périphérique vers l'hôte
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    // Effectuer la réduction finale sur le CPU
    int finalSum = 0;
    for (int i = 0; i < gridSize; i++) {
        finalSum += h_output[i];
    }

    // Afficher le résultat final
    std::cout << "Somme après réduction simplifiée : " << finalSum << std::endl;

    // Libérer la mémoire
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
