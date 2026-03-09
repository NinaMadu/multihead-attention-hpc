#ifndef ATTENTION_H
#define ATTENTION_H

#include <stddef.h>

/* ── GPT-2 small hyper-parameters ─────────────────────────────────── */
#define N_LAYERS   12
#define N_HEADS    12
#define D_MODEL    768
#define D_HEAD     (D_MODEL / N_HEADS)   /* 64   */
#define D_FF       (4 * D_MODEL)         /* 3072 */
#define MAX_SEQ    1024
#define VOCAB_SIZE 50257

/*
 * Weight layout note
 * ------------------
 * GPT-2 uses HuggingFace Conv1D, which stores weights as (in, out).
 * Forward: output = input @ weight + bias
 *
 * In C we access as: output[j] += input[i] * weight[i][j]
 * So the C array dimension order is [in][out], matching Conv1D directly.
 * The binary file stores them in this exact layout — no transpose.
 */

/* ── Per-layer attention weights ───────────────────────────────────── */
typedef struct {
    float c_attn_w[D_MODEL][3 * D_MODEL]; /* [768][2304]  in=D, out=3D  */
    float c_attn_b[3 * D_MODEL];          /* [2304]                     */
    float c_proj_w[D_MODEL][D_MODEL];     /* [768][768]   in=D, out=D   */
    float c_proj_b[D_MODEL];              /* [768]                      */
} AttentionWeights;

/* ── Per-layer MLP weights ─────────────────────────────────────────── */
typedef struct {
    float c_fc_w  [D_MODEL][D_FF];   /* [768][3072]  in=D,  out=4D */
    float c_fc_b  [D_FF];            /* [3072]                      */
    float c_proj_w[D_FF][D_MODEL];   /* [3072][768]  in=4D, out=D  */
    float c_proj_b[D_MODEL];         /* [768]                       */
} MLPWeights;

/* ── Full GPT-2 weight struct ─────────────────────────────────────── */
typedef struct {
    float wte[VOCAB_SIZE][D_MODEL];   /* token embeddings  [50257][768] */
    float wpe[MAX_SEQ][D_MODEL];      /* pos   embeddings  [1024][768]  */

    AttentionWeights attn[N_LAYERS];
    MLPWeights       mlp [N_LAYERS];

    float ln1_g[N_LAYERS][D_MODEL];
    float ln1_b[N_LAYERS][D_MODEL];
    float ln2_g[N_LAYERS][D_MODEL];
    float ln2_b[N_LAYERS][D_MODEL];
    float ln_f_g[D_MODEL];
    float ln_f_b[D_MODEL];
} GPT2Weights;

/* ── Timing result ─────────────────────────────────────────────────── */
typedef struct {
    double serial_ms;
    double parallel_ms;
    double speedup;
} TimingResult;

/* ── Prototypes ────────────────────────────────────────────────────── */
void attention_serial  (const float *input, const AttentionWeights *w,
                        float *output, int seq_len);
void attention_parallel(const float *input, const AttentionWeights *w,
                        float *output, int seq_len);

void layer_norm(float *x, const float *g, const float *b, int n);
void softmax   (float *x, int n);

int  load_weights (GPT2Weights *gpt, const char *path);
int  tokenize     (const char *text, int *tokens, int max_tokens);
void decode_tokens(const int *tokens, int n, char *out, int out_len);

int          forward_pass(GPT2Weights *gpt, const int *tokens,
                          int n_tokens, int use_parallel);
TimingResult benchmark   (GPT2Weights *gpt, const int *tokens,
                          int n_tokens, int n_runs);

#endif /* ATTENTION_H */