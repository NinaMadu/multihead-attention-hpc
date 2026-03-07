#ifndef SERIAL_ATTENTION_H
#define SERIAL_ATTENTION_H

// Matrix dimensions
#define MAX_DIM 512

// Scaled Dot-Product Attention: softmax(Q*K^T / sqrt(d)) * V
void scaled_dot_product_attention(
    float Q[MAX_DIM][MAX_DIM],
    float K[MAX_DIM][MAX_DIM],
    float V[MAX_DIM][MAX_DIM],
    float output[MAX_DIM][MAX_DIM],
    int seq_len,
    int d_k
);

// Multi-Head Attention wrapper
void multi_head_attention(
    float input[MAX_DIM][MAX_DIM],
    float output[MAX_DIM][MAX_DIM],
    int seq_len,
    int d_model,
    int num_heads
);

// Utility functions
void softmax_row(float row[MAX_DIM], int size);
void matrix_multiply(float A[MAX_DIM][MAX_DIM], float B[MAX_DIM][MAX_DIM], 
                     float C[MAX_DIM][MAX_DIM], int n);

#endif