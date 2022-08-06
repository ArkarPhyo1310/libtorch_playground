cc_binary(
    name = "main",
    srcs = ["main.cpp"],
    deps = [
        "@lib_CV//:opencv",
        "@lib_TORCH//:libtorch",
        "@lib_ARGPARSE//:argparse",
        "//Classification/ImageProcessing:ImageProcessor",
        "//Classification/Models:Predictor",
        "//Classification/Utils:File",
    ],
    copts = select({
            "@platforms//os:windows": [
                "/std:c++17", 
            ],
            "@platforms//os:macos": [
                "-std=c++17",
            ],
            "@platforms//os:linux": [
                '-std=c++17',
                '-fPIC',
            ],
        }),
    data = [
        "//Classification:copy"
    ]
)

genrule(
    name = "copy",
    srcs = ["Resources/data"],
    outs = ["data"],
    output_to_bindir = 1,
    cmd = "cp -r $< $@",
    visibility=["//visibility:public"],
)