#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "serial/attention.h"

#define SEQ_LEN 64
#define D_K 64

int main()
{
    // Allocate matrices dynamically
    float (*Q)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));
    float (*K)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));
    float (*V)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));
    float (*output)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));

    if (!Q || !K || !V || !output) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize with random values for testing
    srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < D_K; j++) {
            Q[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            K[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            V[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }

    printf("========================================\n");
    printf("Serial Scaled Dot-Product Attention\n");
    printf("========================================\n");
    printf("Sequence Length: %d\n", SEQ_LEN);
    printf("Head Dimension: %d\n", D_K);

    // Measure execution time
    clock_t start = clock();
    scaled_dot_product_attention(Q, K, V, output, SEQ_LEN, D_K);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Execution Time: %.4f seconds\n", elapsed);
    printf("========================================\n");

    // Print sample output (first 4x4 block)
    printf("\nSample Output (first 4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.6f ", output[i][j]);
        }
        printf("\n");
    }

    printf("\n========================================\n");
    printf("✓ Serial baseline test complete!\n");

    // Free allocated memory
    free(Q);
    free(K);
    free(V);
    free(output);

    return 0;
}