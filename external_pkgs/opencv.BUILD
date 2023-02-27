load("@libtorchPG//:config.bzl", "OPENCV_VERSION")

OPENCV_LIBS = [
    "calib3d",
    "core",
    "dnn",
    "features2d",
    "flann",
    "gapi",
    "highgui",
    "imgcodecs",
    "imgproc",
    "ml",
    "objdetect",
    "photo",
    "stitching",
    "video",
    "videoio",
]

[
    (
        cc_import(
            name = module,
            shared_library = select({
                "@bazel_tools//src/conditions:windows": "bin/opencv_{}{}.dll".format(module, OPENCV_VERSION),
            }),
        )
    )
    for module in OPENCV_LIBS
]

cc_library(
    name="opencv",
    srcs=select({
        "@bazel_tools//src/conditions:windows": glob(["lib/*.lib"]),
        "@bazel_tools//src/conditions:linux": glob(["lib/*.so*"])  
    }),
    hdrs=select({
        "@bazel_tools//src/conditions:windows": glob(["include/**/*.h*"]),
        "@bazel_tools//src/conditions:linux": glob(["include/opencv4/**"])
    }),
    deps=select({
        "@bazel_tools//src/conditions:windows": OPENCV_LIBS,
        "@bazel_tools//src/conditions:linux": []
    }),
    includes=["include", "include/opencv4"],
    visibility=["//visibility:public"],
    linkstatic = 1,
)