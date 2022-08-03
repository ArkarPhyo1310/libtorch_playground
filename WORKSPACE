workspace(name = "classification")

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