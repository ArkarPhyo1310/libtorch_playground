workspace(name = "classification")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@classification//:config.bzl", "OPENCV_PATH", "LIBTORCH_PATH")

new_local_repository(
    name = "lib_CV",
    path = OPENCV_PATH,
    build_file = "@classification//external_pkgs:opencv.BUILD",
)

new_local_repository(
    name = "lib_TORCH",
    path = LIBTORCH_PATH,
    build_file = "@classification//external_pkgs:libtorch.BUILD"
)

http_archive(
    name = "lib_ARGPARSE",
    urls = ["https://github.com/p-ranav/argparse/archive/refs/tags/v2.6.zip"],
    build_file = "@classification//external_pkgs:argparse.BUILD",
    sha256 = "ce4e58d527b83679bdcc4adfa852af7ec9df16b76c11637823ef642cb02d2618",
    strip_prefix = "argparse-2.6",
)