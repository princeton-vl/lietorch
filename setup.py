from setuptools import setup
import os
import sys
import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CppExtension,
    CUDA_HOME,
)

ROOT = os.path.dirname(os.path.abspath(__file__))

def has_cuda_toolkit():
    # Respect FORCE_CUDA=1 to force CUDA build when toolkit is present
    force_cuda = os.getenv("FORCE_CUDA", "0") == "1"
    return (CUDA_HOME is not None) and (force_cuda or torch.cuda.is_available())


use_cuda = has_cuda_toolkit()

# Build list of extensions conditionally
ext_modules = []

if use_cuda:
    # Full CUDA + CPU backend
    ext_modules.append(
        CUDAExtension(
            name="lietorch_backends",
            include_dirs=[
                os.path.join(ROOT, "lietorch/include"),
                os.path.join(ROOT, "eigen"),
            ],
            sources=[
                "lietorch/src/lietorch.cpp",
                "lietorch/src/lietorch_gpu.cu",
                "lietorch/src/lietorch_cpu.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": ["-O2"],
            },
        )
    )

    # Extras are CUDA-only
    ext_modules.append(
        CUDAExtension(
            name="lietorch_extras",
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
            },
        )
    )
else:
    # CPU-only backend: use stub header to avoid CUDA includes
    ext_modules.append(
        CppExtension(
            name="lietorch_backends",
            include_dirs=[
                os.path.join(ROOT, "lietorch/stub_include"),  # must come first
                os.path.join(ROOT, "lietorch/include"),
                os.path.join(ROOT, "eigen"),
            ],
            sources=[
                "lietorch/src/lietorch.cpp",
                "lietorch/src/lietorch_cpu.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O2"],
            },
        )
    )

setup(
    name="lietorch",
    version="0.3",
    description="Lie Groups for PyTorch",
    author="Zachary Teed",
    packages=["lietorch"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
