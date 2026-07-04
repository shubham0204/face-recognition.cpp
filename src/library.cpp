#include "library.h"
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_io.h>

#include <iostream>

void hello() {
    executorch::runtime::EValue eValue;
    executorch::extension::Module module("/path/to/model.pte");
    executorch::runtime::Result<std::vector<executorch::runtime::EValue>> results = module.get_outputs();

    dlib::matrix<dlib::rgb_pixel> rgb;
    dlib::matrix<dlib::rgb_pixel> rgba;
}
