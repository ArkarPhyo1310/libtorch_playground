cc_library(
    name = "YOLOV5PostProcessor",
    srcs = ["PostProcess.cpp"],
    hdrs = ["PostProcess.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//LTPG/Models/Detection/PostProcessor/Abstract:AbstractPostProcessor",
        "@lib_CV//:opencv",
        "@lib_TORCH//:libtorch",
    ],
)
