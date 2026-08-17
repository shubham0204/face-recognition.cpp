#ifndef FACENET_VECTORSEARCH_IMAGEUTILS_H
#define FACENET_VECTORSEARCH_IMAGEUTILS_H
#include "dlib/image_loader/load_image.h"
#include "dlib/matrix/matrix.h"
#include "dlib/pixel.h"

class Utils {
  public:
    static DlibRgbImage loadImageFromFile(const std::string& filePath) {
        DlibRgbImage image;
        dlib::load_image(image, filePath);
        return image;
    }

    static DlibRgbImage loadImageFromIntBuffer(const int* buffer, int width, int height) {
        DlibRgbImage image(width, height);
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                dlib::rgb_pixel pixel;
                pixel.red = (buffer[i * width + j] >> 16) & 0xFF;
                pixel.green = (buffer[i * width + j] >> 8) & 0xFF;
                pixel.blue = (buffer[i * width + j]) & 0xFF;
                image(i, j) = pixel;
            }
        }
        return image;
    }

    static FaceBoundingBox fromDlibRectangle(const dlib::rectangle& rect) {
        FaceBoundingBox boundingBox{};
        boundingBox.left = rect.left();
        boundingBox.top = rect.top();
        boundingBox.bottom = rect.bottom();
        boundingBox.right = rect.right();
        return boundingBox;
    }
};

#endif // FACENET_VECTORSEARCH_IMAGEUTILS_H
