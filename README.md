

This project is an attempt to build an integrated face recognition pipeline that comprises of a face detector (`dlib`), an embedding model (FaceNet-512 - InceptionV3) and an in-memory vector database. 

## Project Structure

### Build Tools

* CMake 4.2 and above for building (build-system generator)
* GCC or Clang toolchains
* Android NDK to build for Android targets
* Python 3.12

### Directories

* `src`: Contains C++ source files and internal headers for target `facenet_vectorsearch`
* `tests`: Contains [GoogleTest](https://github.com/google/googletest) unit tests
* `examples`: End-to-end examples using the C++ library
* `external`: Contains Git sub-modules of external libraries used in the project, i.e. `dlib` and `executorch`
* `include`: Public C++ headers accompanying the `facenet_vectorsearch` library
* `bindings`: Bindings to integrate native code in other managed languages

## Architecture and Design

The face-recognition pipeline contains the following components, invoked on the *hot* path in the same order:

1. Face Detector (`dlib`)
2. Face Embedder (FaceNet-512 executed with ExecuTorch)
3. Vector DB (custom implementation)

The *hot* path is invoked by the `recognize(frameImage)` called on each frame of the camera by the Android app.

Other *cold* paths are:

1. Adding/registering images via the `insert(personName, images)` call
2. Clearing the vector DB or removing images specific to a `personName`

## Motivation

I developed the [OnDevice-Face-Recognition-Android](https://github.com/shubham0204/OnDevice-Face-Recognition-Android) project in 2024 which currently uses ExecuTorch for running the FaceNet 512 model, MLKit for face detection and ObjectBox for vector search capabilities. It also includes spoof-detection via FasNet models and LiteRT. The project itself is a resurrection of an [older project](https://github.com/shubham0204/FaceRecognition_With_FaceNet_Android) I built in 2020.

The latency of the pipeline is good, but I wanted to make it faster and reduce the app's size and memory footprint. 

1. MLKit, ExecuTorch and ObjectBox are written in C++ and have JNI bindings for inclusion in Android app projects.

2. The AAR package of ExecuTorch, available on Maven Central, contains pre-compiled shared libraries (`.so` files) for 64-bit Android targets. Reducing the size of the AAR package is possible by compiling ExecuTorch to only select a set of operators for a backend. In our face-recognition pipeline, the only model we execute is FaceNet 512.

3. ObjectBox is a great embedded database, and one of the few options in Android that support vector search. I wanted to build my own in-memory vector index which is tightly coupled to the face embedding model and leaner.

Considering the above points, I decided to develop an end-to-end face recognition pipeline entirely in C++. Instead of the above three components communicating via JNI and bringing the control flow to/fro ART and native-code (coming from MLKit, ExecuTorch and ObjectBox), we pack the functionality of these three components in a single native codebase and expose it to the Android app via simple functions like `insert(personName, images)` and `recognize(frameImage)`.


## CMake Build Options

1. `BUILD_EXAMPLES`: Controls whether to build the examples present in the `examples` directory
   * Default: `OFF`
2. `BUILD_TESTS`: Controls whether to build unit tests present in the `tests` directory
   * Default: `OFF`
3. `BUILD_JNI_BINDING`: Controls whether to build JNI bindings present in the `bindings/android` directory
   * Default: `OFF`

## Optimizations

* Avoid dynamic memory allocation on the hot-path.
    * Ex: `pixelFloatData` used to store input image pixel data where each pixel is represented as three `int` (R, G and
      B components) is initialized with the class `FaceEmbedder`.
    * Ex: Reserving the size of `std::vector` where the number of elements to be stored are known beforehand.
      `std::vector` is a dynamic array and calling `push_back` may cause reallocation of the entire array. Read
      more [here](https://stackoverflow.com/a/32200517/13546426).
* Reduce executable or shared library size by selectively choosing kernels from ExecuTorch. Check
  the [docs](https://docs.pytorch.org/executorch/stable/kernel-library-selective-build.html). The following CMake
  options enable us to package only the kernels used in the provided `.pte` model:

    ```
    -DEXECUTORCH_SELECT_OPS_MODEL=/Users/shubhampanchal/CLionProjects/facenet-vectorsearch/tests/resources/model.pte \
    -DEXECUTORCH_DTYPE_SELECTIVE_BUILD=ON
    ```
* Using a 4-bit quantized FaceNet model instead of using the full precision `float32` model.

## Vector Database

* Simple in-memory database, stores vector records in a `std::vector` and computes the nearest-neighbor by iterating
  through all records in a traditional `for` loop.
* The vector records need to survive application-restarts, hence we load the records from a file on the disk
  (deserialization) when the database is initialized and save the records to a file (serialization) when the database
  instance is destroyed.
* We choose `flatbuffers` as the serialization format to ensure zero-copy, faster reads/writes from the disk.
* Updates to the vector DB through `insert()`, `remove()` or `clear()` are performed on the in-memory representation. To
  enhance fault tolerance, we write the in-memory representation of the vector index to disk every `10` seconds on a
  different (background) thread to avoid blocking the main thread processing the `recognize` calls.


