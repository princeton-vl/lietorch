from setuptools import setup
import os
import glob
import subprocess
import sys
try:
    import torch
except ImportError:
    raise RuntimeError(
        "lietorch must be built against your installed PyTorch (CUDA/ROCm/CPU build), which is not "
        "visible in pip's isolated build environment. Install PyTorch first and rerun pip with "
        "`--no-build-isolation`.") from None
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, ROCM_HOME

ROOT = os.path.dirname(os.path.abspath(__file__))

# Select the GPU backend from the installed PyTorch build and toolkit:
#   ROCm: torch.version.hip is set and ROCM_HOME (hipcc) is available
#   CUDA: torch.version.cuda is set and CUDA_HOME (nvcc) is available
#   CPU:  otherwise
# Set LIETORCH_FORCE_CPU=1 to force a CPU-only build.
# On ROCm, torch's CUDAExtension hipifies the CUDA sources during the build.
FORCE_CPU = os.environ.get("LIETORCH_FORCE_CPU", "0") == "1"
if FORCE_CPU:
    BACKEND = "cpu"
elif getattr(torch.version, "hip", None) is not None and ROCM_HOME is not None:
    BACKEND = "rocm"
elif torch.version.cuda is not None and CUDA_HOME is not None:
    BACKEND = "cuda"
else:
    BACKEND = "cpu"
WITH_GPU = BACKEND in ("cuda", "rocm")
print(f"lietorch: building with {BACKEND.upper()} backend")

include_dirs = [
    os.path.join(ROOT, "lietorch/include"),
    os.path.join(ROOT, "eigen"),
]
sources = [
    "lietorch/src/lietorch.cpp",
    "lietorch/src/lietorch_cpu.cpp",
]

# Package indexes searched (in order) for rocm-sdk-devel; override with LIETORCH_ROCM_INDEX_URL.
ROCM_INDEX_URLS = [
    "https://repo.amd.com/rocm/whl/gfx120X-all/",
    "https://rocm.prereleases.amd.com/whl/gfx120X-all/",
    "https://rocm.nightlies.amd.com/v2/gfx120X-all/",
    "https://stable.repo.amd.com/rocm/whl-next/",
]


def fetch_rocm_headers():
    """Download the rocm-sdk-devel wheel matching the installed ROCm SDK and extract only
    the thrust/rocprim headers into build/rocm_headers (the environment is left untouched)."""
    import tarfile
    import tempfile
    import zipfile
    from importlib import metadata

    target = os.path.join(ROOT, "build", "rocm_headers")
    if os.path.isfile(os.path.join(target, "thrust", "complex.h")):
        return target

    try:
        version = metadata.version("rocm-sdk-core")
    except metadata.PackageNotFoundError:
        version = torch.version.hip.split("-")[0]
    major, minor = version.split(".")[:2]
    spec = f">={major}.{minor}"
    indexes = [u for u in [os.environ.get("LIETORCH_ROCM_INDEX_URL")] if u] or ROCM_INDEX_URLS

    with tempfile.TemporaryDirectory() as tmp:
        wheel = None
        for url in indexes:
            print(f"lietorch: downloading rocm-sdk-devel{spec} from {url}")
            res = subprocess.run(
                [sys.executable, "-m", "pip", "download", f"rocm-sdk-devel{spec}", "--pre",
                 "--no-deps", "--index-url", url, "-d", tmp, "--quiet"])
            wheel = next(iter(glob.glob(os.path.join(tmp, "rocm_sdk_devel-*.whl"))), None)
            if res.returncode == 0 and wheel:
                break
        if not wheel:
            raise RuntimeError(
                f"ROCm build requires the thrust headers (thrust/complex.h) but rocm-sdk-devel{spec} "
                "could not be downloaded. Install the ROCm development files matching your PyTorch "
                "(`pip install rocm[devel]` from the same index as torch), install a system ROCm with "
                "rocthrust, or set LIETORCH_ROCM_INDEX_URL to a package index providing rocm-sdk-devel.")
        with zipfile.ZipFile(wheel) as z:
            z.extract("rocm_sdk_devel/_devel.tar", tmp)
        prefix = "_rocm_sdk_devel/include/"
        with tarfile.open(os.path.join(tmp, "rocm_sdk_devel", "_devel.tar")) as t:
            members = [m for m in t.getmembers()
                       if m.name.startswith((prefix + "thrust/", prefix + "rocprim/"))]
            for m in members:
                m.name = m.name[len(prefix):]
            os.makedirs(target, exist_ok=True)
            t.extractall(target, members=members)
    return target


def rocm_build_paths():
    """Extra include/library dirs needed by pip-installed ROCm SDKs, which ship
    neither thrust headers (rocm-sdk-devel) nor an unversioned libamdhip64.so."""
    include_dirs, library_dirs = [], []
    roots = [ROCM_HOME]
    # rocm-sdk-devel unpacks itself lazily on the first `rocm-sdk path` call
    try:
        out = subprocess.run([sys.executable, "-m", "rocm_sdk", "path", "--root"],
                             capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            roots.append(out.stdout.strip().splitlines()[-1])
    except OSError:
        pass
    inc = [os.path.join(r, "include") for r in roots if r]
    inc = [d for d in inc if os.path.isfile(os.path.join(d, "thrust", "complex.h"))]
    if not inc:
        inc = [fetch_rocm_headers()]
    include_dirs.append(inc[0])

    lib_dir = os.path.join(ROCM_HOME, "lib") if ROCM_HOME else None
    if lib_dir and not os.path.exists(os.path.join(lib_dir, "libamdhip64.so")):
        versioned = sorted(glob.glob(os.path.join(lib_dir, "libamdhip64.so.*")))
        if versioned:
            link_dir = os.path.join(ROOT, "build", "rocm_links")
            os.makedirs(link_dir, exist_ok=True)
            link = os.path.join(link_dir, "libamdhip64.so")
            if os.path.lexists(link):
                os.remove(link)
            os.symlink(versioned[0], link)
            library_dirs.append(link_dir)
    return include_dirs, library_dirs


if WITH_GPU:
    rocm_includes, rocm_libs = rocm_build_paths() if BACKEND == "rocm" else ([], [])
    include_dirs = include_dirs + rocm_includes
    # PyTorch exposes ROCm devices as torch.device("cuda"), so WITH_CUDA enables
    # the GPU dispatch in both cases; WITH_ROCM additionally marks HIP builds.
    gpu_flags = ["-O2", "-DWITH_CUDA"] + (["-DWITH_ROCM"] if BACKEND == "rocm" else [])
    ext_modules = [
        CUDAExtension("lietorch_backends",
            include_dirs=include_dirs,
            library_dirs=rocm_libs,
            sources=sources + ["lietorch/src/lietorch_gpu.cu"],
            extra_compile_args={
                "cxx": gpu_flags,
                "nvcc": gpu_flags,  # compiled with hipcc on ROCm
            }),

        CUDAExtension("lietorch_extras",
            include_dirs=include_dirs,
            library_dirs=rocm_libs,
            sources=[
                "lietorch/extras/altcorr_kernel.cu",
                "lietorch/extras/corr_index_kernel.cu",
                "lietorch/extras/se3_builder.cu",
                "lietorch/extras/se3_inplace_builder.cu",
                "lietorch/extras/se3_solver.cu",
                "lietorch/extras/extras.cpp",
            ],
            extra_compile_args={
                "cxx": gpu_flags,
                "nvcc": gpu_flags,
            }),
    ]
else:
    ext_modules = [
        CppExtension("lietorch_backends",
            include_dirs=include_dirs,
            sources=sources,
            extra_compile_args={"cxx": ["-O2"]}),
    ]

setup(
    name="lietorch",
    version="0.3",
    description="Lie Groups for PyTorch",
    author="Zachary Teed",
    packages=["lietorch"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
