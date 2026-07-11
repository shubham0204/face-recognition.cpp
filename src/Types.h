#ifndef FACENET_VECTORSEARCH_TYPES_H
#define FACENET_VECTORSEARCH_TYPES_H
#include <array>

#define EMBEDDING_DIM 512
#define INPUT_IMG_DIM 160
using Embedding = std::array<double, EMBEDDING_DIM>;

#endif // FACENET_VECTORSEARCH_TYPES_H
