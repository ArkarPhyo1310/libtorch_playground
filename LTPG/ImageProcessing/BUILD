cc_library(
    name = "ImageProcessor",
    srcs = ["ImageProcessor.cpp"],
    hdrs = ["ImageProcessor.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//LTPG/Utils:Convert",
        "//LTPG/Utils:DataTypes",
        "//LTPG/Utils:Logger",
        "@lib_CV//:opencv",
        "@lib_TORCH//:libtorch",
    ],
)
