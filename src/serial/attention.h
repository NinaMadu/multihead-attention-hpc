#ifndef SERIAL_ATTENTION_H
#define SERIAL_ATTENTION_H

// Matrix dimensions
#define MAX_DIM 512

// Allocates Q, K, V internally
void attention_complete_flow(
    float X[MAX_DIM][MAX_DIM],           // Input tokens (n x d)
    float Wq[MAX_DIM][MAX_DIM],          // Query weight matrix (d x d)
    float Wk[MAX_DIM][MAX_DIM],          // Key weight matrix (d x d)
    float Wv[MAX_DIM][MAX_DIM],          // Value weight matrix (d x d)
    float output[MAX_DIM][MAX_DIM],      // Output (n x d)
    int seq_len,                          // n (sequence length)
    int d_k                               // d (embedding dimension)
);

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
void matrix_multiply_rect(float A[MAX_DIM][MAX_DIM], float B[MAX_DIM][MAX_DIM],
                          float C[MAX_DIM][MAX_DIM], int rows_A, int cols_A, int cols_B);

#endif