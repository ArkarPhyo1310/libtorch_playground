cc_library(
    name = "Detector",
    srcs = ["Detector.cpp"],
    hdrs = ["Detector.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//LTPG/Models/Detection/PostProcessor/Abstract:AbstractPostProcessor",
        "//LTPG/Models/Detection/PostProcessor/YoloV5:YOLOV5PostProcessor",
        "//LTPG/Models/Detection/Utils:DetUtils",
        "@lib_CV//:opencv",
        "@lib_TORCH//:libtorch",
    ],
)
