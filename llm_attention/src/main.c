/*
 * main.c — GPT-2 serial vs parallel attention+MLP benchmark
 *
 * Usage:
 *   ./llm_attn [options] "prompt"
 *   -w <path>   weights file   (default: weights/gpt2.bin)
 *   -r <n>      benchmark runs (default: 5)
 *   -g <n>      tokens to gen  (default: 20)
 *   -t <n>      OMP threads
 *   -s          serial only
 *   -p          parallel only
 */

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "attention.h"

static void print_results(TimingResult r, int bench_seq, int n_threads) {
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║        FULL FORWARD PASS BENCHMARK RESULTS               ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Model       : GPT-2 Small (124M)                        ║\n");
    printf("║  Architecture: %2d layers × %2d heads × d=%d              ║\n",
           N_LAYERS, N_HEADS, D_MODEL);
    printf("║  Bench seq   : %3d tokens (realistic context length)     ║\n", bench_seq);
    printf("║  OMP threads : %3d                                        ║\n", n_threads);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Parallelised: Attention (heads) + MLP (output features) ║\n");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Serial   : %8.2f ms / forward pass                   ║\n", r.serial_ms);
    printf("║  Parallel : %8.2f ms / forward pass                   ║\n", r.parallel_ms);
    printf("║  Speedup  : %8.2f×                                     ║\n", r.speedup);
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    if (r.speedup < 2.0) {
        printf("  Tip: speedup scales with thread count and sequence length.\n");
        printf("  Try:  OMP_NUM_THREADS=12 ./llm_attn -r 3 -g 5 \"your prompt\"\n");
        printf("  Or a longer prompt for more tokens in the benchmark context.\n\n");
    }
}

static void generate(GPT2Weights *gpt, int *ctx, int n_ctx,
                     int max_new, int use_parallel, const char *label) {
    printf("\n── %s generation ──────────────────────────────────────\n", label);

    int tokens[MAX_SEQ];
    memcpy(tokens, ctx, n_ctx * sizeof(int));
    int n = n_ctx;

    char buf[2048];
    decode_tokens(tokens, n, buf, sizeof(buf));
    printf("Prompt : \"%s\"\nOutput :", buf);
    fflush(stdout);

    for (int step = 0; step < max_new && n < MAX_SEQ; step++) {
        int next = forward_pass(gpt, tokens, n, use_parallel);
        tokens[n++] = next;
        decode_tokens(&next, 1, buf, sizeof(buf));
        printf("%s", buf);
        fflush(stdout);
        if (next == 50256) break;
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    const char *weights_path = "weights/gpt2.bin";
    int  n_runs     = 5;
    int  n_generate = 20;
    int  n_threads  = omp_get_max_threads();
    int  do_serial  = 1, do_parallel = 1;
    const char *prompt = "Hi how are you?";

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i],"-w") && i+1<argc) weights_path = argv[++i];
        else if (!strcmp(argv[i],"-r") && i+1<argc) n_runs       = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-g") && i+1<argc) n_generate   = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-t") && i+1<argc) n_threads    = atoi(argv[++i]);
        else if (!strcmp(argv[i],"-s"))              { do_serial=1; do_parallel=0; }
        else if (!strcmp(argv[i],"-p"))              { do_serial=0; do_parallel=1; }
        else if (!strcmp(argv[i],"-h")) {
            printf("Usage: %s [options] [\"prompt\"]\n"
                   "  -w <path>  weights (default: weights/gpt2.bin)\n"
                   "  -r <n>     benchmark runs (default: 5)\n"
                   "  -g <n>     tokens to generate (default: 20)\n"
                   "  -t <n>     OMP threads\n"
                   "  -s / -p    serial-only / parallel-only\n", argv[0]);
            return 0;
        } else prompt = argv[i];
    }

    omp_set_num_threads(n_threads);

    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   GPT-2 Serial vs Parallel Inference Benchmark           ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("Weights : %s\nPrompt  : \"%s\"\n"
           "Runs    : %d  |  Generate: %d tokens  |  Threads: %d/%d\n\n",
           weights_path, prompt, n_runs, n_generate, n_threads,
           omp_get_max_threads());

    printf("Allocating GPT-2 weight struct (~%.0f MB) …\n",
           sizeof(GPT2Weights)/1e6);
    GPT2Weights *gpt = calloc(1, sizeof(GPT2Weights));
    if (!gpt) { fprintf(stderr,"OOM\n"); return 1; }

    if (load_weights(gpt, weights_path) != 0) {
        fprintf(stderr, "\nERROR: Cannot load '%s'.\nRun: python3 download_weights.py\n",
                weights_path);
        free(gpt); return 1;
    }

    int tokens[MAX_SEQ];
    int n_tokens = tokenize(prompt, tokens, MAX_SEQ);
    printf("\nTokenised \"%s\" → %d tokens: ", prompt, n_tokens);
    for (int i = 0; i < n_tokens; i++) printf("%d ", tokens[i]);
    printf("\n");

    if (do_serial)   generate(gpt, tokens, n_tokens, n_generate, 0, "SERIAL");
    if (do_parallel) generate(gpt, tokens, n_tokens, n_generate, 1, "PARALLEL");

    if (do_serial && do_parallel) {
        printf("\nBenchmarking full forward pass (%d runs) …\n", n_runs);
        TimingResult r = benchmark(gpt, tokens, n_tokens, n_runs);
        print_results(r, 32, n_threads);
    }

    free(gpt);
    return 0;
}