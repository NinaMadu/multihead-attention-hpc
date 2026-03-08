#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stddef.h>

#define MAX_WORD_LEN 64
#define MAX_VOCAB 1000   
#define EMBEDDING_DIM 64 

void preprocess_text(char *text);

int load_vocabulary(const char *filename, char vocab[][MAX_WORD_LEN], int *vocab_size);

int token_to_index(const char *token, char vocab[][MAX_WORD_LEN], int vocab_size);

void get_embedding_vector(int index, float embedding[EMBEDDING_DIM]);

void initialize_embedding_matrix(float embedding_matrix[MAX_VOCAB][EMBEDDING_DIM], int vocab_size);

#endif