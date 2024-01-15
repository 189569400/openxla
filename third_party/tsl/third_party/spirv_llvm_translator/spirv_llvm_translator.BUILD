exports_files(["LICENSE"])

cc_library(
    name = "spirv_llvm_translator",
    srcs = glob([
        "lib/SPIRV/libSPIRV/*.cpp",
        "lib/SPIRV/libSPIRV/*.hpp",
        "lib/SPIRV/libSPIRV/*.h",
        "lib/SPIRV/Mangler/*.cpp",
        "lib/SPIRV/Mangler/*.h",
        "lib/SPIRV/*.cpp",
        "lib/SPIRV/*.hpp",
        "lib/SPIRV/*.h",
    ]),
    hdrs = glob(["include/*"]),
    includes = [
        "include/",
        "lib/SPIRV/",
        "lib/SPIRV/Mangler/",
        "lib/SPIRV/libSPIRV/",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:BitWriter",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Demangle",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:Linker",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:TransformUtils",
        "@spirv_headers//:spirv_cpp_headers",
    ],
)
