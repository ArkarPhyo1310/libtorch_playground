TORCH_LIBS = [
    "asmjit",
    "c10",
    "fbgemm",
    "libiomp5md",
    "libiompstubs5md",
    "torch",
    "torch_cpu",
    "torch_global_deps",
    "uv"
]

[
    (
        cc_import(
            name = module,
            shared_library = select({
                "@bazel_tools//src/conditions:windows": "lib/{}.dll".format(module),
            }),
        )
    )
    for module in TORCH_LIBS
]

cc_library(
    name="libtorch",
    srcs=select({
        "@bazel_tools//src/conditions:windows": glob(["lib/*.lib"]),
        "@bazel_tools//src/conditions:linux": glob(["lib/lib*.so*"], exclude=["lib/libtorch_python.so", "lib/libnnapi_backend.so"])  
    }),
    hdrs=glob([
            "include/**",
            "include/ATen/**",
            "include/c10/**",
            "include/caffe2/**",
            "include/torch/**",
            "include/torch/csrc/**",
            "include/torch/csrc/jit/**",
            "include/torch/csrc/api/include/**",
        ]),
    deps=select({
        "@bazel_tools//src/conditions:windows": TORCH_LIBS,
        "@bazel_tools//src/conditions:linux": []
    }), 
    includes=[
        "include",
        "include/torch/csrc/api/include",
    ],
    linkstatic = 1,
    visibility=["//visibility:public"],
)