#ifndef FACENET_VECTORSEARCH_HELPER_H
#define FACENET_VECTORSEARCH_HELPER_H
#include <string>

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

#endif // FACENET_VECTORSEARCH_HELPER_H
