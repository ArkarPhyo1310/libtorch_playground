cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    copts = select({
        "@platforms//os:windows": [
            "/std:c++17",
        ],
        "@platforms//os:macos": [
            "-std=c++17",
        ],
        "@platforms//os:linux": [
            "-std=c++17",
            "-fPIC",
        ],
    }),
    data = [
        "//LTPG:copy",
    ],
    deps = [
        "//LTPG/ImageProcessing:ImageProcessor",
        "//LTPG/Models/Classification:Classifier",
        "//LTPG/Models/Detection:Detector",
        "//LTPG/Utils:File",
        "@lib_ARGPARSE//:argparse",
        "@lib_CV//:opencv",
        "@lib_TORCH//:libtorch",
    ],
)

genrule(
    name = "copy",
    srcs = ["Resources/data"],
    outs = ["data"],
    cmd = "cp -r $< $@",
    output_to_bindir = 1,
    visibility = ["//visibility:public"],
)
