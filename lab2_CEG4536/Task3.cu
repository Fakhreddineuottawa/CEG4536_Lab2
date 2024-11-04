#include <cuda_runtime.h>
#include <iostream>

__global__ void dynamicReductionKernel(int* input, int* output, int size) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x * 2 + tid;

    // Charger les données dans la mémoire partagée, avec vérification des limites
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

__global__ void finalReductionKernel(int* output, int numBlocks) {
    extern __shared__ int sharedData[];
    int tid = threadIdx.x;

    // Charger les données dans la mémoire partagée pour la réduction finale
    if (tid < numBlocks) {
        sharedData[tid] = output[tid];
    }
    else {
        sharedData[tid] = 0;
    }
    __syncthreads();

    // Effectuer la réduction finale
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Écrire la somme finale dans la sortie
    if (tid == 0) {
        output[0] = sharedData[0];
    }
}

int main() {
    const int size = 1024; // Nombre total d'éléments
    const int bytes = size * sizeof(int);
    const int blockSize = 256; // Nombre de threads par bloc
    const int gridSize = (size + blockSize * 2 - 1) / (blockSize * 2); // Nombre de blocs

    int* h_input = new int[size];
    int* h_output = new int[1]; // Besoin d'un seul résultat ici
    int* d_input, * d_output;

    // Initialiser le tableau d'entrée avec des 1 (ou d'autres valeurs selon les instructions du labo)
    for (int i = 0; i < size; i++) {
        h_input[i] = 1;
    }

    // Allouer la mémoire sur le périphérique
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, gridSize * sizeof(int));
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Lancer le noyau de réduction initial avec allocation de mémoire partagée
    dynamicReductionKernel << <gridSize, blockSize, blockSize * sizeof(int) >> > (d_input, d_output, size);
    cudaDeviceSynchronize();

    // Si plus d'un bloc a été utilisé, lancer le noyau de réduction finale
    if (gridSize > 1) {
        finalReductionKernel << <1, blockSize, blockSize * sizeof(int) >> > (d_output, gridSize);
        cudaDeviceSynchronize();
    }

    // Copier le résultat de retour vers l'hôte
    cudaMemcpy(h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);

    // Afficher le résultat final
    std::cout << "Somme après réduction parallèle dynamique : " << h_output[0] << std::endl;

    // Libérer la mémoire
    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
