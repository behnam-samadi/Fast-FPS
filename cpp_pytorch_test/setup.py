from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CppExtension

setup(
    name='cppcuda_tutorial',
    version='1.0',
    description='cppcuda_tutorial',
    long_description='cppcuda_tutorial',
    ext_modules=[
        CppExtension(
            name='cppcuda_tutorial',
            sources=['utils.cpp']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)