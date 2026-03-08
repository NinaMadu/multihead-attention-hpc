#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "embedding.h"

int main(void) {
    srand(42);

    // Load vocabulary
    char vocab[MAX_VOCAB][MAX_WORD_LEN];
    int vocab_size = 0;
    load_vocabulary("../../data/vocab.txt", vocab, &vocab_size);

    // Initialize embedding matrix
    float embedding_matrix[MAX_VOCAB][EMBEDDING_DIM];
    initialize_embedding_matrix(embedding_matrix, vocab_size);

    // Read input text
    char *input = NULL;
    size_t len = 0;
    printf("Enter your text input: ");
    getline(&input, &len, stdin);
    if (!input) {
        return 1;
    }
    if (strlen(input) > 0 && input[strlen(input) - 1] == '\n') {
        input[strlen(input) - 1] = '\0';
    }

    // Preprocess text
    preprocess_text(input);

    // Tokenize input
    int seq_len = 0;
    char *tokens[128];
    char *token = strtok(input, " ");
    while (token) {
        tokens[seq_len++] = token;
        token = strtok(NULL, " ");
    }

    if (seq_len == 0) {
        printf("No tokens found in input.\n");
        free(input);
        return 0;
    }

    // Convert tokens → embedding vectors
    float (*X)[EMBEDDING_DIM] = malloc(seq_len * sizeof(float[EMBEDDING_DIM]));
    for (int i = 0; i < seq_len; i++) {
        int idx = token_to_index(tokens[i], vocab, vocab_size);
        get_embedding_vector(idx, X[i]);
    }

    // Example: print first token vector
    for (int j = 0; j < EMBEDDING_DIM; j++) printf("%.4f ", X[0][j]);
    printf("\n");

    free(input);
    free(X);
    return 0;
}