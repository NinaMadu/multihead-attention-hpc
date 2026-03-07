#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "serial/attention.h"

#define NUM_RUNS 3

void benchmark_attention(int seq_len, int d_k) {
    // Allocate matrices
    float (*Q)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));
    float (*K)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));
    float (*V)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));
    float (*output)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));

    // Initialize
    srand(42);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_k; j++) {
            Q[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            K[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            V[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }

    // Run benchmarks
    double total_time = 0.0;
    for (int run = 0; run < NUM_RUNS; run++) {
        clock_t start = clock();
        scaled_dot_product_attention(Q, K, V, output, seq_len, d_k);
        clock_t end = clock();
        total_time += (double)(end - start) / CLOCKS_PER_SEC;
    }

    double avg_time = total_time / NUM_RUNS;
    printf("SeqLen: %4d, D_K: %4d, Avg Time: %.6f seconds\n", seq_len, d_k, avg_time);

    free(Q);
    free(K);
    free(V);
    free(output);
}

int main() {
    printf("========================================\n");
    printf("Serial Attention Benchmark\n");
    printf("========================================\n\n");

    int sizes[] = {16, 32, 64, 128, 256, 512};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        benchmark_attention(sizes[i], sizes[i]);
    }

    printf("\n========================================\n");
    printf("Benchmark complete!\n");

    return 0;
}