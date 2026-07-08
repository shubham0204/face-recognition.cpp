#include "FaceRecognizer.h"

int main(int argc, char* argv[]) {
    const std::string dbFilePath = "/Users/shubhampanchal/CLionProjects/facenet-vectorsearch/tests/resources/db.bin";
    const std::string modelFilePath = "/Users/shubhampanchal/CLionProjects/facenet-vectorsearch/tests/resources/model.pte";
    const std::string imagePath = "/Users/shubhampanchal/CLionProjects/facenet-vectorsearch/tests/resources/img.png";
    const auto faceRecognizer = createFaceRecognizer(dbFilePath, modelFilePath);
    faceRecognizer->insert("John Doe", {imagePath});
    const auto results = faceRecognizer->recognize(imagePath);
    return 0;
}
