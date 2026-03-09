/*
 * attention_serial.c — single-threaded multi-head scaled dot-product attention
 *
 * Weight layout: Conv1D stores as [in, out].
 * Matrix multiply: output[j] = bias[j] + sum_i( input[i] * weight[i][j] )
 *
 *   Step 1: [Q|K|V] = input · c_attn_w + c_attn_b
 *           input:    [seq × D_MODEL]
 *           c_attn_w: [D_MODEL][3·D_MODEL]   (in=D, out=3D)
 *           result:   [seq × 3·D_MODEL]
 *
 *   Step 2: Per-head scaled dot-product attention
 *           scores = Q_h · K_h^T / sqrt(d_k)  + causal mask
 *           softmax → weighted sum over V_h
 *
 *   Step 3: output = attn_concat · c_proj_w + c_proj_b
 *           c_proj_w: [D_MODEL][D_MODEL]      (in=D, out=D)
 */

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "attention.h"

void softmax(float *x, int n) {
    float mv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mv) mv = x[i];
    float s = 0.f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mv); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

void layer_norm(float *x, const float *g, const float *b, int n) {
    float mean = 0.f;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    float var = 0.f;
    for (int i = 0; i < n; i++) { float d = x[i]-mean; var += d*d; }
    var /= n;
    float inv = 1.f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++)
        x[i] = g[i] * (x[i] - mean) * inv + b[i];
}

void attention_serial(const float *input, const AttentionWeights *w,
                      float *output, int seq_len) {

    float *Q        = malloc(seq_len * D_MODEL * sizeof(float));
    float *K        = malloc(seq_len * D_MODEL * sizeof(float));
    float *V        = malloc(seq_len * D_MODEL * sizeof(float));
    float *attn_out = malloc(seq_len * D_MODEL * sizeof(float));
    float *scores   = malloc(seq_len * seq_len  * sizeof(float));
    memset(attn_out, 0, seq_len * D_MODEL * sizeof(float));

    /* ── Step 1: QKV projection ────────────────────────────────────
     * c_attn_w is [D_MODEL][3*D_MODEL] (in=row, out=col)
     * output[t][j] = bias[j] + sum_i( input[t][i] * c_attn_w[i][j] )
     */
    for (int t = 0; t < seq_len; t++) {
        const float *xt = input + t * D_MODEL;
        for (int j = 0; j < 3 * D_MODEL; j++) {
            float acc = w->c_attn_b[j];
            for (int i = 0; i < D_MODEL; i++)
                acc += xt[i] * w->c_attn_w[i][j];
            /* split into Q / K / V */
            if      (j < D_MODEL)         Q[t*D_MODEL + j]              = acc;
            else if (j < 2*D_MODEL)       K[t*D_MODEL + (j -   D_MODEL)] = acc;
            else                          V[t*D_MODEL + (j - 2*D_MODEL)] = acc;
        }
    }

    /* ── Step 2: Per-head scaled dot-product attention ─────────────── */
    float scale = 1.f / sqrtf((float)D_HEAD);

    for (int h = 0; h < N_HEADS; h++) {
        int off = h * D_HEAD;

        /* scores[t1,t2] = Q_h[t1] · K_h[t2] / sqrt(d_k) */
        for (int t1 = 0; t1 < seq_len; t1++) {
            for (int t2 = 0; t2 <= t1; t2++) {
                float dot = 0.f;
                for (int d = 0; d < D_HEAD; d++)
                    dot += Q[t1*D_MODEL+off+d] * K[t2*D_MODEL+off+d];
                scores[t1*seq_len+t2] = dot * scale;
            }
            for (int t2 = t1+1; t2 < seq_len; t2++)
                scores[t1*seq_len+t2] = -1e9f;
            softmax(scores + t1*seq_len, seq_len);
        }

        /* attn_out[t1,off+d] = sum_t2( scores[t1,t2] * V_h[t2,d] ) */
        for (int t1 = 0; t1 < seq_len; t1++)
            for (int d = 0; d < D_HEAD; d++) {
                float acc = 0.f;
                for (int t2 = 0; t2 < seq_len; t2++)
                    acc += scores[t1*seq_len+t2] * V[t2*D_MODEL+off+d];
                attn_out[t1*D_MODEL+off+d] = acc;
            }
    }

    /* ── Step 3: Output projection ─────────────────────────────────
     * c_proj_w is [D_MODEL][D_MODEL] (in=row, out=col)
     * output[t][j] = bias[j] + sum_i( attn_out[t][i] * c_proj_w[i][j] )
     */
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