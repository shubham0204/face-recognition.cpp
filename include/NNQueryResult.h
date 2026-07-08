#ifndef FACENET_VECTORSEARCH_NNQUERYRESUKT_H
#define FACENET_VECTORSEARCH_NNQUERYRESUKT_H
#include "FaceBoundingBox.h"

#include <string>

struct NNQueryResult {
    std::string personName;
    float cosineSimilarity = 0.0f;
    FaceBoundingBox faceBoundingBox;
};

#endif // FACENET_VECTORSEARCH_NNQUERYRESUKT_H
