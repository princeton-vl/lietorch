from setuptools import setup
import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

setup(
    name="lietorch",
    version="0.3",
    description="Lie Groups for PyTorch",
    author="Zachary Teed",
    packages=["lietorch"],
    ext_modules=[
        CUDAExtension("lietorch_backends", 
            include_dirs=[
                os.path.join(ROOT, "lietorch/include"), 
                os.path.join(ROOT, "eigen")],
            sources=[
                "lietorch/src/lietorch.cpp", 
                "lietorch/src/lietorch_gpu.cu",
                "lietorch/src/lietorch_cpu.cpp"],
            extra_compile_args={
                "cxx": ["-O2"], 
                "nvcc": ["-O2"],
            }),

        CUDAExtension("lietorch_extras", 
            sources=[
                "lietorch/extras/altcorr_kernel.cu",
                "lietorch/extras/corr_index_kernel.cu",
                "lietorch/extras/se3_builder.cu",
                "lietorch/extras/se3_inplace_builder.cu",
                "lietorch/extras/se3_solver.cu",
                "lietorch/extras/extras.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2"], 
                "nvcc": ["-O2"],
            }),
    ],
    cmdclass={ "build_ext": BuildExtension }
)
