#ifndef FACENET_VECTORSEARCH_TYPES_H
#define FACENET_VECTORSEARCH_TYPES_H
#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"

#include <array>

#define EMBEDDING_DIM 512
#define INPUT_IMG_DIM 160
using Embedding = std::array<double, EMBEDDING_DIM>;
using DlibRgbImage = dlib::matrix<dlib::rgb_pixel>;

#endif // FACENET_VECTORSEARCH_TYPES_H
