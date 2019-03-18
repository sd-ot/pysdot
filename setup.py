#!/usr/bin/env python3

from setuptools import setup, find_packages, Extension

sd3d = Extension(
    'pybind_sdot_3d_double',
    sources=['pysdot/cpp/pybind_sdot.cpp'],
    include_dirs=['ext/pybind11/include'],
    define_macros=[
        ('PD_MODULE_NAME', 'pybind_sdot_3d_double'),
        ('PD_TYPE', 'double'),
        ('PD_DIM', '3')
    ],
    extra_compile_args=['-march=native', '-ffast-math'],
)

setup(
    name='pysdot',
    version='0.1',
    packages=find_packages(exclude=[
        'hugo', 'ext', 'build', 'dist',
        'examples', 'results', 'tests'
    ]),
    ext_modules=[sd3d],
    install_requires=[
        "numpy",
    ],
)
