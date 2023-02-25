workspace(name = "libtorchPG")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@libtorchPG//:config.bzl", "LIBTORCH_PATH", "OPENCV_PATH")

new_local_repository(
    name = "lib_CV",
    build_file = "@libtorchPG//external_pkgs:opencv.BUILD",
    path = OPENCV_PATH,
)

new_local_repository(
    name = "lib_TORCH",
    build_file = "@libtorchPG//external_pkgs:libtorch.BUILD",
    path = LIBTORCH_PATH,
)

http_archive(
    name = "lib_ARGPARSE",
    build_file = "@libtorchPG//external_pkgs:argparse.BUILD",
    sha256 = "55396ae05d9deb8030b8ad9babf096be9c35652d5822d8321021bcabb25f4b72",
    strip_prefix = "argparse-2.9",
    urls = ["https://github.com/p-ranav/argparse/archive/refs/tags/v2.9.zip"],
)
