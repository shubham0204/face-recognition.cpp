#ifndef FACENET_VECTORSEARCH_HELPER_H
#define FACENET_VECTORSEARCH_HELPER_H
#include <string>
#include <filesystem>
#include "FaceEmbedder.h"

/**
 *  Accepts a resource `filename` and returns the absolute path.
 *  This function is used in tests to load resources like images.
 *  TEST_RESOURCE_DIR is defined in tests/CMakeLists.txt as a compiler definition
 *  for this test target.
 *
 * @param filename Name of the file within the tests/resources directory
 * @return Absolute path of the resource
 */
inline std::string getResourcePath(const std::string& filename) { return std::string(TEST_RESOURCE_DIR) + filename; }

inline void deleteResource(const std::string& filename) {
    std::filesystem::remove(getResourcePath(filename));
}

inline Embedding createRandomEmbedding() {
    Embedding embedding{};
    for (int i = 0; i < EMBEDDING_DIM; i++) {
        embedding[i] = drand48() / static_cast<double>(RAND_MAX);
    }
    return embedding;
}

#endif // FACENET_VECTORSEARCH_HELPER_H
