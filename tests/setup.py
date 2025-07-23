from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


root = Path(__file__).parent

setup(
    name="cublas_ext",
    ext_modules=[
        CUDAExtension(
            name="cublas_ext",
            sources=[str(root / "cublas_ext.cu")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)

