/*
 * tokenizer.c  (v3 – fixed strdup / C11 compatibility)
 * -----------------------------------------------------
 * Root cause of segfault: strdup() is not declared under -std=c11
 * (it requires _POSIX_C_SOURCE >= 200809L).  The compiler silently
 * treated it as returning int, producing garbage pointers that crashed
 * on first dereference.
 *
 * Fix: replace strdup() with our own str_dup() that is always available.
 * Fix: add #define _POSIX_C_SOURCE before system headers as belt-and-braces.
 */

#define _POSIX_C_SOURCE 200809L   /* enables strdup, popen, etc. in glibc */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "attention.h"

/* ── portable strdup (avoids the C11 implicit-declaration problem) ── */
static char *str_dup(const char *s) {
    size_t len = strlen(s) + 1;
    char  *p   = malloc(len);
    if (p) memcpy(p, s, len);
    return p;
}

/* ── open-addressing hash table: token_id → UTF-8 string ─────────── */
#define HTAB_SIZE  (1 << 17)   /* 131072 slots – load ~0.38 for 50257 entries */
#define HTAB_MASK  (HTAB_SIZE - 1)

typedef struct { int id; char *text; } HEntry;
static HEntry htab[HTAB_SIZE];
static int    htab_loaded = 0;

static void htab_insert(int id, const char *text) {
    unsigned h = (unsigned)id & HTAB_MASK;
    while (htab[h].text && htab[h].id != id)
        h = (h + 1) & HTAB_MASK;
    if (!htab[h].text) {
        htab[h].id   = id;
        htab[h].text = str_dup(text);   /* ← was strdup(), now str_dup() */
    }
}

static const char *htab_get(int id) {
    unsigned h = (unsigned)id & HTAB_MASK;
    while (htab[h].text) {
        if (htab[h].id == id) return htab[h].text;
        h = (h + 1) & HTAB_MASK;
    }
    return NULL;
}

/* ── unescape \n \t \\ written by tokenize_prompt.py ─────────────── */
static void unescape(const char *src, char *dst, int dlen) {
    int j = 0;
    for (int i = 0; src[i] && j < dlen - 1; i++) {
        if (src[i] == '\\' && src[i + 1]) {
            i++;
            switch (src[i]) {
                case 'n':  dst[j++] = '\n'; break;
                case 't':  dst[j++] = '\t'; break;
                case '\\': dst[j++] = '\\'; break;
                default:   dst[j++] = '\\'; dst[j++] = src[i]; break;
            }
        } else {
            dst[j++] = src[i];
        }
    }
    dst[j] = '\0';
}

/* ── load decode_table.txt produced by tokenize_prompt.py ────────── */
static int load_decode_table(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[tokenizer] Cannot open decode table: %s\n", path);
        return -1;
    }

    char line[4096], unesc[2048];
    int  count = 0;

    while (fgets(line, sizeof(line), f)) {
        char *tab = strchr(line, '\t');
        if (!tab) continue;
        *tab = '\0';
        int   id   = atoi(line);
        char *text = tab + 1;

        /* strip trailing CR/LF */
        int tl = (int)strlen(text);
        while (tl > 0 && (text[tl-1] == '\n' || text[tl-1] == '\r'))
            text[--tl] = '\0';

        unescape(text, unesc, sizeof(unesc));
        htab_insert(id, unesc);
        count++;
    }
    fclose(f);
    printf("[tokenizer] Decode table loaded: %d entries\n", count);
    return count > 0 ? 0 : -1;
}

/* ── public: tokenize ─────────────────────────────────────────────── */
int tokenize(const char *text, int *tokens, int max_tokens) {
    const char *tok_bin = "/tmp/gpt2_tokens.bin";
    const char *dec_tab = "/tmp/gpt2_decode_table.txt";

    /* Escape any double-quotes in the prompt */
    char safe[1024];
    int  k = 0;
    for (const char *p = text; *p && k < 1020; p++) {
        if (*p == '"') { safe[k++] = '\\'; safe[k++] = '"'; }
        else             safe[k++] = *p;
    }
    safe[k] = '\0';

    char cmd[4096];
    snprintf(cmd, sizeof(cmd),
        "python3 tokenize_prompt.py \"%s\" \"%s\" \"%s\"",
        safe, tok_bin, dec_tab);

    printf("[tokenizer] Running: %s\n", cmd);
    int rc = system(cmd);
    if (rc != 0) {
        fprintf(stderr,
            "[tokenizer] ERROR: tokenize_prompt.py failed (rc=%d)\n"
            "  Make sure you are in the project root and venv is active:\n"
            "    source venv/bin/activate\n"
            "    pip install transformers\n", rc);
        return 0;
    }

    /* Read binary token file */
    FILE *f = fopen(tok_bin, "rb");
    if (!f) {
        fprintf(stderr, "[tokenizer] Cannot open %s\n", tok_bin);
        return 0;
    }

    uint32_t n = 0;
    if (fread(&n, 4, 1, f) != 1) { fclose(f); return 0; }
    if ((int)n > max_tokens) n = (uint32_t)max_tokens;

    for (uint32_t i = 0; i < n; i++) {
        int32_t id = 0;
        if (fread(&id, 4, 1, f) != 1) { fclose(f); return (int)i; }
        tokens[i] = (int)id;
    }
    fclose(f);

    /* Load decode table exactly once */
    if (!htab_loaded) {
        if (load_decode_table(dec_tab) == 0)
            htab_loaded = 1;
        else
            fprintf(stderr, "[tokenizer] Warning: decode table not loaded — output will show token IDs\n");
    }

    return (int)n;
}

/* ── public: decode_tokens ────────────────────────────────────────── */
void decode_tokens(const int *tokens, int n, char *out, int out_len) {
    /* Lazy-load the decode table if tokenize() was not called first */
    if (!htab_loaded) {
        if (load_decode_table("/tmp/gpt2_decode_table.txt") == 0)
            htab_loaded = 1;
    }

    char *ptr = out;
    int   rem = out_len - 1;

    for (int i = 0; i < n && rem > 0; i++) {
        const char *txt = htab_get(tokens[i]);
        if (txt) {
            int tl   = (int)strlen(txt);
            int copy = tl < rem ? tl : rem;
            memcpy(ptr, txt, copy);
            ptr += copy;
            rem -= copy;
        } else {
            /* Fallback: raw byte for low IDs, else [tok_N] */
            int w = (tokens[i] >= 0 && tokens[i] < 256)
                  ? snprintf(ptr, rem + 1, "%c", (char)tokens[i])
                  : snprintf(ptr, rem + 1, "[tok_%d]", tokens[i]);
            if (w > 0) { ptr += w; rem -= w; }
        }
    }
    *ptr = '\0';
}