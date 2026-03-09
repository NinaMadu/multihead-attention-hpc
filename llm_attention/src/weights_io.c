/*
 * weights_io.c  — load GPT-2 weights from binary file
 * -----------------------------------------------------
 * Binary layout produced by download_weights.py (v2):
 *
 *   [magic  uint32 = 0x47505432]
 *   [n_layers uint32]
 *   For each layer:
 *     attn.c_attn_w  [D_MODEL × 3·D_MODEL]
 *     attn.c_attn_b  [3·D_MODEL]
 *     attn.c_proj_w  [D_MODEL × D_MODEL]
 *     attn.c_proj_b  [D_MODEL]
 *     mlp.c_fc_w     [D_MODEL × D_FF]
 *     mlp.c_fc_b     [D_FF]
 *     mlp.c_proj_w   [D_FF × D_MODEL]
 *     mlp.c_proj_b   [D_MODEL]
 *   wte  [VOCAB_SIZE × D_MODEL]
 *   wpe  [MAX_SEQ × D_MODEL]
 *   For each layer:
 *     ln1_g [D_MODEL]
 *     ln1_b [D_MODEL]
 *     ln2_g [D_MODEL]
 *     ln2_b [D_MODEL]
 *   ln_f_g [D_MODEL]
 *   ln_f_b [D_MODEL]
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "attention.h"

#define MAGIC 0x47505432u

static int rf(FILE *f, float *dst, size_t count) {
    size_t got = fread(dst, sizeof(float), count, f);
    if (got != count) {
        fprintf(stderr, "[weights_io] Short read: wanted %zu got %zu\n",
                count, got);
        return -1;
    }
    return 0;
}

int load_weights(GPT2Weights *gpt, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }

    unsigned int magic = 0, n_layers = 0;
    if (fread(&magic,    4, 1, f) != 1 ||
        fread(&n_layers, 4, 1, f) != 1) {
        fprintf(stderr, "[weights_io] Cannot read header\n");
        fclose(f); return -1;
    }

    if (magic != MAGIC) {
        fprintf(stderr, "[weights_io] Bad magic 0x%08X — re-run download_weights.py\n",
                magic);
        fclose(f); return -1;
    }
    if ((int)n_layers != N_LAYERS) {
        fprintf(stderr, "[weights_io] Layer mismatch: file=%u code=%d\n",
                n_layers, N_LAYERS);
        fclose(f); return -1;
    }

    printf("[weights_io] Loading %u layers …\n", n_layers);

    for (int l = 0; l < N_LAYERS; l++) {
        AttentionWeights *aw = &gpt->attn[l];
        MLPWeights       *mw = &gpt->mlp[l];

        /* Attention */
        if (rf(f, &aw->c_attn_w[0][0], (size_t)D_MODEL * 3 * D_MODEL)) goto err;
        if (rf(f, aw->c_attn_b,        (size_t)3 * D_MODEL))            goto err;
        if (rf(f, &aw->c_proj_w[0][0], (size_t)D_MODEL * D_MODEL))      goto err;
        if (rf(f, aw->c_proj_b,        (size_t)D_MODEL))                 goto err;

        /* MLP */
        if (rf(f, &mw->c_fc_w  [0][0], (size_t)D_MODEL * D_FF))         goto err;
        if (rf(f, mw->c_fc_b,          (size_t)D_FF))                    goto err;
        if (rf(f, &mw->c_proj_w[0][0], (size_t)D_FF * D_MODEL))         goto err;
        if (rf(f, mw->c_proj_b,        (size_t)D_MODEL))                 goto err;

        printf("[weights_io]   layer %2d ✓\n", l);
    }

    /* Embeddings */
    if (rf(f, &gpt->wte[0][0], (size_t)VOCAB_SIZE * D_MODEL)) goto err;
    if (rf(f, &gpt->wpe[0][0], (size_t)MAX_SEQ    * D_MODEL)) goto err;
    printf("[weights_io] Embeddings ✓\n");

    /* Layer norms */
    for (int l = 0; l < N_LAYERS; l++) {
        if (rf(f, gpt->ln1_g[l], D_MODEL)) goto err;
        if (rf(f, gpt->ln1_b[l], D_MODEL)) goto err;
        if (rf(f, gpt->ln2_g[l], D_MODEL)) goto err;
        if (rf(f, gpt->ln2_b[l], D_MODEL)) goto err;
    }
    if (rf(f, gpt->ln_f_g, D_MODEL)) goto err;
    if (rf(f, gpt->ln_f_b, D_MODEL)) goto err;
    printf("[weights_io] Layer norms ✓\n");

    fclose(f);
    printf("[weights_io] All weights loaded successfully.\n");
    return 0;

err:
    fclose(f); return -1;
}