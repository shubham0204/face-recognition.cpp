

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

1. `BUILD_EXAMPLES`:
   * Default: `OFF`
2. `BUILD_TESTS`:
   * Default: `OFF`
3. `BUILD_JNI_BINDING`:
   * Default: `OFF`


