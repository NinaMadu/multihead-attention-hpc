/*
 * attention_parallel.c — OpenMP multi-head attention
 *
 * Same math as attention_serial.c with three OpenMP parallel regions:
 *   1. QKV projection: parallel over tokens (t)
 *   2. Per-head attention: parallel over heads (h) — main speedup
 *   3. Output projection: parallel over tokens (t)
 *
 * No race conditions: each head h writes to disjoint slice
 *   attn_out[t][h*D_HEAD .. (h+1)*D_HEAD-1]
 * and uses its own private score buffer:
 *   scores + h * seq_len * seq_len
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "attention.h"

void attention_parallel(const float *input, const AttentionWeights *w,
                        float *output, int seq_len) {

    float *Q        = malloc(seq_len * D_MODEL * sizeof(float));
    float *K        = malloc(seq_len * D_MODEL * sizeof(float));
    float *V        = malloc(seq_len * D_MODEL * sizeof(float));
    float *attn_out = malloc(seq_len * D_MODEL * sizeof(float));
    /* N_HEADS private score buffers to avoid any shared writes */
    float *scores   = malloc((size_t)N_HEADS * seq_len * seq_len * sizeof(float));
    memset(attn_out, 0, seq_len * D_MODEL * sizeof(float));

    /* ── Step 1: Parallel QKV projection (over tokens) ──────────── */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seq_len; t++) {
        const float *xt = input + t * D_MODEL;
        for (int j = 0; j < 3 * D_MODEL; j++) {
            float acc = w->c_attn_b[j];
            for (int i = 0; i < D_MODEL; i++)
                acc += xt[i] * w->c_attn_w[i][j];
            if      (j < D_MODEL)         Q[t*D_MODEL + j]               = acc;
            else if (j < 2*D_MODEL)       K[t*D_MODEL + (j -   D_MODEL)] = acc;
            else                          V[t*D_MODEL + (j - 2*D_MODEL)]  = acc;
        }
    }

    /* ── Step 2: Parallel per-head attention (over heads) ──────────
     * Each head h is fully independent:
     *   - Private score buffer: scores + h * seq_len * seq_len
     *   - Writes to disjoint output slice: off = h * D_HEAD
     */
    float scale = 1.f / sqrtf((float)D_HEAD);

    #pragma omp parallel for schedule(static)
    for (int h = 0; h < N_HEADS; h++) {
        int    off  = h * D_HEAD;
        float *sc_h = scores + (size_t)h * seq_len * seq_len;

        /* Scores + causal mask */
        for (int t1 = 0; t1 < seq_len; t1++) {
            for (int t2 = 0; t2 <= t1; t2++) {
                float dot = 0.f;
                for (int d = 0; d < D_HEAD; d++)
                    dot += Q[t1*D_MODEL+off+d] * K[t2*D_MODEL+off+d];
                sc_h[t1*seq_len+t2] = dot * scale;
            }
            for (int t2 = t1+1; t2 < seq_len; t2++)
                sc_h[t1*seq_len+t2] = -1e9f;

            /* Inline softmax (avoids function call overhead) */
            float *row = sc_h + t1*seq_len;
            float  mv  = row[0];
            for (int t2 = 1; t2 < seq_len; t2++) if (row[t2]>mv) mv=row[t2];
            float s = 0.f;
            for (int t2 = 0; t2 < seq_len; t2++) { row[t2]=expf(row[t2]-mv); s+=row[t2]; }
            for (int t2 = 0; t2 < seq_len; t2++) row[t2] /= s;
        }

        /* Weighted sum — writes to disjoint head slice, no race condition */
        for (int t1 = 0; t1 < seq_len; t1++)
            for (int d = 0; d < D_HEAD; d++) {
                float acc = 0.f;
                for (int t2 = 0; t2 < seq_len; t2++)
                    acc += sc_h[t1*seq_len+t2] * V[t2*D_MODEL+off+d];
                attn_out[t1*D_MODEL+off+d] = acc;   /* disjoint: no race */
            }
    }

    /* ── Step 3: Parallel output projection (over tokens) ───────── */
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < seq_len; t++) {
        const float *at = attn_out + t*D_MODEL;
        float       *ot = output   + t*D_MODEL;
        for (int j = 0; j < D_MODEL; j++) {
            float acc = w->c_proj_b[j];
            for (int i = 0; i < D_MODEL; i++)
                acc += at[i] * w->c_proj_w[i][j];
            ot[j] = acc;
        }
    }

    free(Q); free(K); free(V); free(attn_out); free(scores);
}