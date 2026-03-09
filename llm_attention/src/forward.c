/*
 * forward.c — complete GPT-2 forward pass with full parallelism
 * --------------------------------------------------------------
 * Parallel regions:
 *   Attention: QKV proj (over t), heads (over h), out proj (over t)
 *   MLP:       c_fc expand (over j in D_FF), c_proj contract (over j in D_MODEL),
 *              and the token loop (over t)
 *
 * The benchmark times the FULL forward pass (attn + MLP), not just attention,
 * so the reported speedup reflects real inference speedup.
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "attention.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static int argmax(const float *v, int n) {
    int best = 0; float bv = v[0];
    for (int i = 1; i < n; i++) if (v[i] > bv) { bv = v[i]; best = i; }
    return best;
}

/* GELU: 0.5 * x * (1 + tanh(sqrt(2/π)*(x + 0.044715*x³))) */
static inline float gelu(float x) {
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.f + tanhf(inner));
}

/* ── Serial MLP (single token) ───────────────────────────────────── */
static void mlp_token_serial(const float *x, const MLPWeights *m, float *out) {
    float *h = malloc(D_FF * sizeof(float));

    for (int j = 0; j < D_FF; j++) {
        float acc = m->c_fc_b[j];
        for (int i = 0; i < D_MODEL; i++) acc += x[i] * m->c_fc_w[i][j];
        h[j] = gelu(acc);
    }
    for (int j = 0; j < D_MODEL; j++) {
        float acc = m->c_proj_b[j];
        for (int i = 0; i < D_FF; i++) acc += h[i] * m->c_proj_w[i][j];
        out[j] = acc;
    }
    free(h);
}

/* ── Parallel MLP (all tokens, OpenMP over j in inner loops) ─────── */
static void mlp_parallel(const float *x_all, const MLPWeights *m,
                          float *out_all, int seq_len) {
    /*
     * Parallelise two ways simultaneously:
     *   a) outer token loop: each thread handles different tokens
     *   b) inner feature loop: parallelise j within each token's computation
     *
     * Strategy: collapse token × output-feature loops.
     * Each (t, j) pair is independent.
     */

    /* Step 1: expand to hidden [seq_len × D_FF] — parallel over (t, j) */
    float *h = malloc(seq_len * D_FF * sizeof(float));

    #pragma omp parallel for schedule(static) collapse(2)
    for (int t = 0; t < seq_len; t++) {
        for (int j = 0; j < D_FF; j++) {
            const float *xt = x_all + t * D_MODEL;
            float acc = m->c_fc_b[j];
            for (int i = 0; i < D_MODEL; i++) acc += xt[i] * m->c_fc_w[i][j];
            h[t * D_FF + j] = gelu(acc);
        }
    }

    /* Step 2: contract to output [seq_len × D_MODEL] — parallel over (t, j) */
    #pragma omp parallel for schedule(static) collapse(2)
    for (int t = 0; t < seq_len; t++) {
        for (int j = 0; j < D_MODEL; j++) {
            const float *ht = h + t * D_FF;
            float acc = m->c_proj_b[j];
            for (int i = 0; i < D_FF; i++) acc += ht[i] * m->c_proj_w[i][j];
            out_all[t * D_MODEL + j] = acc;
        }
    }

    free(h);
}

/* ── Full forward pass ───────────────────────────────────────────── */
int forward_pass(GPT2Weights *gpt, const int *tokens, int n_tokens,
                 int use_parallel) {

    float *x = calloc(n_tokens * D_MODEL, sizeof(float));
    float *h = calloc(n_tokens * D_MODEL, sizeof(float));
    if (!x || !h) { fprintf(stderr, "OOM\n"); exit(1); }

    /* 1. Embeddings */
    for (int t = 0; t < n_tokens; t++) {
        float *xt = x + t * D_MODEL;
        for (int d = 0; d < D_MODEL; d++)
            xt[d] = gpt->wte[tokens[t]][d] + gpt->wpe[t][d];
    }

    /* 2. Transformer layers */
    for (int l = 0; l < N_LAYERS; l++) {

        /* ── Attention sub-layer ───────────────────────────────── */
        memcpy(h, x, n_tokens * D_MODEL * sizeof(float));
        for (int t = 0; t < n_tokens; t++)
            layer_norm(h + t*D_MODEL, gpt->ln1_g[l], gpt->ln1_b[l], D_MODEL);

        float *attn_out = calloc(n_tokens * D_MODEL, sizeof(float));
        if (!attn_out) { fprintf(stderr,"OOM\n"); exit(1); }

        if (use_parallel)
            attention_parallel(h, &gpt->attn[l], attn_out, n_tokens);
        else
            attention_serial  (h, &gpt->attn[l], attn_out, n_tokens);

        for (int i = 0; i < n_tokens * D_MODEL; i++) x[i] += attn_out[i];
        free(attn_out);

        /* ── MLP sub-layer ─────────────────────────────────────── */
        memcpy(h, x, n_tokens * D_MODEL * sizeof(float));
        for (int t = 0; t < n_tokens; t++)
            layer_norm(h + t*D_MODEL, gpt->ln2_g[l], gpt->ln2_b[l], D_MODEL);

        float *mlp_out = calloc(n_tokens * D_MODEL, sizeof(float));
        if (!mlp_out) { fprintf(stderr,"OOM\n"); exit(1); }

        if (use_parallel) {
            mlp_parallel(h, &gpt->mlp[l], mlp_out, n_tokens);
        } else {
            for (int t = 0; t < n_tokens; t++)
                mlp_token_serial(h + t*D_MODEL, &gpt->mlp[l], mlp_out + t*D_MODEL);
        }

        for (int i = 0; i < n_tokens * D_MODEL; i++) x[i] += mlp_out[i];
        free(mlp_out);
    }

    /* 3. Final layer norm */
    for (int t = 0; t < n_tokens; t++)
        layer_norm(x + t*D_MODEL, gpt->ln_f_g, gpt->ln_f_b, D_MODEL);

    /* 4. Logits: last token × wte^T */
    float *last_h = x + (n_tokens - 1) * D_MODEL;
    float *logits = malloc(VOCAB_SIZE * sizeof(float));
    for (int v = 0; v < VOCAB_SIZE; v++) {
        float acc = 0.f;
        for (int d = 0; d < D_MODEL; d++) acc += last_h[d] * gpt->wte[v][d];
        logits[v] = acc;
    }

    int next = argmax(logits, VOCAB_SIZE);
    free(logits); free(x); free(h);
    return next;
}

/* ── Benchmark: full forward pass, growing context ───────────────── */
/*
 * We benchmark with the FULL context length that actually occurs during
 * generation (prompt + generated tokens), not just the 5-token prompt.
 * This gives a realistic speedup figure.
 *
 * bench_seq_len: number of tokens to use for the timing test.
 * We use the prompt tokens padded/repeated to bench_seq_len so the
 * workload is representative.
 */
TimingResult benchmark(GPT2Weights *gpt, const int *tokens,
                       int n_tokens, int n_runs) {
    TimingResult r = {0};

    /* Use a realistic context: 32 tokens (prompt repeated to fill) */
    int bench_len = 32;
    int bench_tokens[1024];
    for (int i = 0; i < bench_len; i++)
        bench_tokens[i] = tokens[i % n_tokens];

    printf("Running SERIAL   (%d runs, seq=%d) …\n", n_runs, bench_len);
    double t0 = now_ms();
    for (int i = 0; i < n_runs; i++)
        forward_pass(gpt, bench_tokens, bench_len, 0);
    r.serial_ms = (now_ms() - t0) / n_runs;

    printf("Running PARALLEL (%d runs, seq=%d) …\n", n_runs, bench_len);
    t0 = now_ms();
    for (int i = 0; i < n_runs; i++)
        forward_pass(gpt, bench_tokens, bench_len, 1);
    r.parallel_ms = (now_ms() - t0) / n_runs;

    r.speedup = r.serial_ms / r.parallel_ms;
    return r;
}