#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "attention.h"

// Helper: Softmax for a single row
void softmax_row(float row[MAX_DIM], int size)
{
    float max_val = row[0];
    for (int i = 1; i < size; i++) {
        if (row[i] > max_val) max_val = row[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        row[i] = expf(row[i] - max_val);  // Numerical stability
        sum += row[i];
    }

    for (int i = 0; i < size; i++) {
        row[i] /= sum;
    }
}

// Helper: Matrix multiplication for rectangular matrices
// C = A * B where A is (rows_A x cols_A) and B is (cols_A x cols_B)
void matrix_multiply_rect(float A[MAX_DIM][MAX_DIM], float B[MAX_DIM][MAX_DIM],
                          float C[MAX_DIM][MAX_DIM], int rows_A, int cols_A, int cols_B)
{
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < cols_A; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Helper: Matrix multiplication C = A * B
void matrix_multiply(float A[MAX_DIM][MAX_DIM], float B[MAX_DIM][MAX_DIM],
                     float C[MAX_DIM][MAX_DIM], int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Complete Attention Flow: Input X -> Q,K,V -> Score Matrix -> Output
void attention_complete_flow(
    float X[MAX_DIM][MAX_DIM],           // Input tokens (n x d)
    float Wq[MAX_DIM][MAX_DIM],          // Query weight matrix (d x d)
    float Wk[MAX_DIM][MAX_DIM],          // Key weight matrix (d x d)
    float Wv[MAX_DIM][MAX_DIM],          // Value weight matrix (d x d)
    float output[MAX_DIM][MAX_DIM],      // Output (n x d)
    int seq_len,                          // n (sequence length)
    int d_k)                              // d (embedding dimension)
{
    // Allocate Q, K, V dynamically to avoid stack overflow
    float (*Q)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));
    float (*K)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));
    float (*V)[MAX_DIM] = malloc(MAX_DIM * sizeof(float[MAX_DIM]));

    if (!Q || !K || !V) {
        printf("Memory allocation failed in attention_complete_flow!\n");
        return;
    }

    printf("\n========== ATTENTION COMPLETE FLOW ==========\n");
    printf("Step 1: Compute Q = X * Wq (size: %d x %d)\n", seq_len, d_k);
    matrix_multiply_rect(X, Wq, Q, seq_len, d_k, d_k);
    printf("✓ Q computed\n");

    printf("\nStep 2: Compute K = X * Wk (size: %d x %d)\n", seq_len, d_k);
    matrix_multiply_rect(X, Wk, K, seq_len, d_k, d_k);
    printf("✓ K computed\n");

    printf("\nStep 3: Compute V = X * Wv (size: %d x %d)\n", seq_len, d_k);
    matrix_multiply_rect(X, Wv, V, seq_len, d_k, d_k);
    printf("✓ V computed\n");

    printf("\nStep 4: Compute Score Matrix = Q * K^T (size: %d x %d)\n", seq_len, seq_len);
    
    printf("\nStep 5: Compute Attention Output\n");
    scaled_dot_product_attention(Q, K, V, output, seq_len, d_k);
    printf("✓ Attention computed\n");

    printf("\nStep 6: Output (size: %d x %d)\n", seq_len, d_k);
    printf("✓ Complete flow finished!\n");
    printf("==========================================\n\n");

    // Free temporary matrices
    free(Q);
    free(K);
    free(V);
}

// Scaled Dot-Product Attention
void scaled_dot_product_attention(
    float Q[MAX_DIM][MAX_DIM],
    float K[MAX_DIM][MAX_DIM],
    float V[MAX_DIM][MAX_DIM],
    float output[MAX_DIM][MAX_DIM],
    int seq_len,
    int d_k)
{
    float scores[MAX_DIM][MAX_DIM];

    // Step 1: Compute Q * K^T
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            scores[i][j] = 0.0f;
            for (int k = 0; k < d_k; k++) {
                scores[i][j] += Q[i][k] * K[j][k];  // K^T means K[j][k]
            }
        }
    }

    // Step 2: Scale by sqrt(d_k)
    float scale = 1.0f / sqrtf((float)d_k);
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            scores[i][j] *= scale;
        }
    }

    // Step 3: Apply softmax to each row
    for (int i = 0; i < seq_len; i++) {
        softmax_row(scores[i], seq_len);
    }

    // Step 4: Multiply attention weights by V
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < d_k; j++) {
            output[i][j] = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                output[i][j] += scores[i][k] * V[k][j];
            }
        }
    }
}

// Multi-Head Attention wrapper (placeholder for now)
void multi_head_attention(
    float input[MAX_DIM][MAX_DIM],
    float output[MAX_DIM][MAX_DIM],
    int seq_len,
    int d_model,
    int num_heads)
{
    printf("Multi-head attention not yet implemented\n");
}