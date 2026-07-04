# ============= example_simple executable target configuration ======================
add_executable(example_simple main.cpp)
target_link_libraries(
        example_simple
        PRIVATE
        facenet_vectorsearch
)