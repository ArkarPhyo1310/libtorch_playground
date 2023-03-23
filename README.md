# <div align="center"> Libtorch Playground </div>

## <div align="center">  :snowflake: Inferencing with Pytorch C++ API (libtorch) :snowflake: </div>

This project is aimed for Windows OS. But it should work on Linux environment with teeny tiny changes.

### :skull_and_crossbones: Installation :skull_and_crossbones

1. C++17 (must have)
2. **Bazel**        - For more [info](https://bazel.build/install).
3. **OpenCV**       - For more [info](https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html).
4. **LibTorch**     - For more [info](https://pytorch.org/cppdocs/installing.html).

For installing these packages, please refer to this [page](https://github.com/ArkarPhyo1310/DevEnv4Windows).

>Note: If you encounter "undefiend error" in linux, try downloading the nightly version of the libtorch without pre-compile.
>Here is the link to download. [https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip]

### :package: External Packages :package

- [argparse](https://github.com/p-ranav/argparse) [![Stars](https://img.shields.io/github/stars/p-ranav/argparse.svg?style=flat&logo=GitHub&logoColor=white&color=blue)](https://github.com/p-ranav/argparse)

- [spdlog](https://github.com/gabime/spdlog) [![Stars](https://img.shields.io/github/stars/gabime/spdlog.svg?style=flat&logo=GitHub&logoColor=white&color=blue)](https://github.com/gabime/spdlog)

### :clipboard: Configuration :clipboard

Current **config.bzl**.

```cmd
OPENCV_PATH = "D:/Dev_pkgs/opencv/install" # Add your OPENCV Path Here
OPENCV_VERSION = "460" # linux 4.6.0
LIBTORCH_PATH = "D:/Dev_pkgs/pytorch/libtorch" # Add your LIBTORCH Path here
```

Please update the path accordingly.

### :zap: Usage :zap

```cmd
bazel run //LTPG:main 
```

Above command will build and run with default image and model.

For more command-line options,

For Linux,

```cmd
bazel-bin/LTPG/main --help
```

For Windows,

```cmd
./bazel-bin/LTPG/main.exe --help
```

Feedback and suggestions are all welcomed. :v:
