#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "serial/attention.h"

#define SEQ_LEN 64
#define D_K 64

int main()
{
    // Allocate matrices dynamically
    float (*X)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));        // Input tokens
    float (*Wq)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));      // Query weights
    float (*Wk)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));      // Key weights
    float (*Wv)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));      // Value weights
    float (*output)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));  // Output

    if (!X || !Wq || !Wk || !Wv || !output) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Initialize with random values for testing
    srand(42);  // Fixed seed for reproducibility
    
    printf("========================================\n");
    printf("Serial Complete Attention Flow Test\n");
    printf("========================================\n");
    printf("Sequence Length (n): %d\n", SEQ_LEN);
    printf("Embedding Dimension (d): %d\n", D_K);
    
    // Initialize Input Tokens X (n x d)
    printf("\nInitializing Input Tokens X (%d x %d)...\n", SEQ_LEN, D_K);
    for (int i = 0; i < SEQ_LEN; i++) {
        for (int j = 0; j < D_K; j++) {
            X[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
    printf("✓ X initialized\n");

    // Initialize Weight Matrices (d x d each)
    printf("\nInitializing Weight Matrices Wq, Wk, Wv (%d x %d each)...\n", D_K, D_K);
    for (int i = 0; i < D_K; i++) {
        for (int j = 0; j < D_K; j++) {
            Wq[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            Wk[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
            Wv[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
    printf("✓ Wq, Wk, Wv initialized\n");

    // Measure execution time for complete flow
    clock_t start = clock();
    attention_complete_flow(X, Wq, Wk, Wv, output, SEQ_LEN, D_K);
    clock_t end = clock();

    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\nTotal Execution Time: %.4f seconds\n", elapsed);
    printf("========================================\n");

    // Print sample output (first 4x4 block)
    printf("\nSample Output (first 4x4):\n");
    for (int i = 0; i < 4 && i < SEQ_LEN; i++) {
        for (int j = 0; j < 4 && j < D_K; j++) {
            printf("%.6f ", output[i][j]);
        }
        printf("\n");
    }

    printf("\n========================================\n");
    printf("✓ Serial complete flow test finished!\n");

    // Free allocated memory
    free(X);
    free(Wq);
    free(Wk);
    free(Wv);
    free(output);

    return 0;
}