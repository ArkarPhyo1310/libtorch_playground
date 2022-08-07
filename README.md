# <div align="center"> Libtorch Classification </div>

## <div align="center">  :snowflake: Inferencing with Pytorch C++ API (libtorch) :snowflake: </div>

This project is aimed for Windows OS. But it should work on Linux environment with teeny tiny changes.

### :skull_and_crossbones: Installation :skull_and_crossbones:

1. **Bazel**        - For more [info](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html).
2. **OpenCV**       - For more [info](https://bazel.build/install).
3. **LibTorch**     - For more [info](https://pytorch.org/cppdocs/installing.html).

For installing these packages, please refer to this [page](https://github.com/ArkarPhyo1310/DevEnv4Windows).

### :clipboard: Configuration :clipboard:

Current **config.bzl**.

```
OPENCV_PATH = "D:/Dev_pkgs/opencv/install" # Add your OPENCV Path Here
OPENCV_VERSION = "460" # linux 4.6.0
LIBTORCH_PATH = "D:/Dev_pkgs/pytorch/libtorch" # Add your LIBTORCH Path here
```
Please update the path accordingly.

### :zap: Usage :zap:

```cmd
bazel run //Classification:main 
```

Above command will build and run with default image and model (resnet18).

For more command-line options, 

```
./bazel-bin/Classification/main.exe --help
```

Feedback and suggestions are all welcomed. :v: