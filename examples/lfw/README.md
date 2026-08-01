# Examples: LFW (Labelled Faces in the Wild) Insertion and Search

Create a directory named `data`:

```shell
cd examples/lfw
mkdir data
```

Download the LFW dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data) and place the
following files in the specified structure:

```text
data ->
    lfw-deepfunneled ->
        Aaron_Eckhart ->
        Aaron_Guiel ->
        ...
    matchpairsDevTrain.csv
    model.pte
insertion.cpp
search.cpp
Utils.h
CMakeLists.txt
README.md
```

Build the project and execute the `example_lfw_insertion` and `example_lfw_search` binaries:

```shell
./example_lfw_insertion
./example_lfw_search
1012
1100
```

The numbers `1012` and `1100` indicate the number of correct predictions and total samples present in
`matchpairsDevTrain.csv`.