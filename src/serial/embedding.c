#include "embedding.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Internal global embedding matrix
static float internal_embedding_matrix[MAX_VOCAB][EMBEDDING_DIM];

void preprocess_text(char *text) {
    int i = 0, j = 0;

    while (text[i]) {
        char c = text[i];        
        if (c >= 'A' && c <= 'Z') {
            c = c + 32;
        }        
        if (isalnum(c) || c == ' ') {
            text[j++] = c;
        }
        i++;
    }
    text[j] = '\0';
}

// Load vocabulary from file
int load_vocabulary(const char *filename, char vocab[][MAX_WORD_LEN], int *vocab_size) {
    FILE *fp = fopen(filename, "r");
    if (!fp) return -1;

    char line[MAX_WORD_LEN];
    int count = 0;
    while (fgets(line, sizeof(line), fp) && count < MAX_VOCAB) {
        line[strcspn(line, "\n")] = '\0'; 
        strcpy(vocab[count], line);
        count++;
    }
    *vocab_size = count;
    fclose(fp);
    return 0;
}

// Convert token to index
int token_to_index(const char *token, char vocab[][MAX_WORD_LEN], int vocab_size) {
    for (int i = 0; i < vocab_size; i++) {
        if (strcmp(token, vocab[i]) == 0) return i;
    }
    return -1; 
}

// Initialize embedding matrix randomly
void initialize_embedding_matrix(float embedding_matrix[MAX_VOCAB][EMBEDDING_DIM], int vocab_size) {
    for (int i = 0; i < vocab_size; i++) {
        for (int j = 0; j < EMBEDDING_DIM; j++) {
            float value = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // -1 to 1
            embedding_matrix[i][j] = value;
            internal_embedding_matrix[i][j] = value;
        }
    }
}

// Get embedding vector for an index
void get_embedding_vector(int index, float embedding[EMBEDDING_DIM]) {
    for (int j = 0; j < EMBEDDING_DIM; j++) {
        embedding[j] = internal_embedding_matrix[index][j];
    }
}