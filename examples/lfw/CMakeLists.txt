# ============= example_lfw executables target configuration ======================
add_executable(example_lfw_insertion insertion.cpp Utils.h)
target_link_libraries(
        example_lfw_insertion
        PRIVATE
        facenet_vectorsearch
)
add_executable(example_lfw_search search.cpp Utils.h)

target_compile_definitions(
        example_lfw_insertion
        PRIVATE
        SRC_DIR="${CMAKE_SOURCE_DIR}/examples/lfw/"
)
target_compile_definitions(
        example_lfw_search
        PRIVATE
        SRC_DIR="${CMAKE_SOURCE_DIR}/examples/lfw/"
)
target_link_libraries(
        example_lfw_search
        PRIVATE
        facenet_vectorsearch
)