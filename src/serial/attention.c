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
    float attention_weights[MAX_DIM][MAX_DIM];

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