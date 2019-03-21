#!/usr/bin/env python3

from setuptools import setup, find_packages, Extension

ext_modules = []
for TF in ["double"]:
    for dim in [2, 3]:
        name = 'pybind_sdot_{}d_{}'.format(dim, TF)
        ext_modules.append(Extension(
            name,
            sources=['pysdot/cpp/pybind_sdot.cpp'],
            include_dirs=['ext/pybind11/include'],
            define_macros=[
                ('PD_MODULE_NAME', name),
                ('PD_TYPE', TF),
                ('PD_DIM', str(dim))
            ],
            language='c++',
            extra_compile_args=['-march=native', '-ffast-math'],
        ))

setup(
    name='pysdot',
    version='0.1',
    packages=find_packages(exclude=[
        'hugo', 'ext', 'build', 'dist',
        'examples', 'results', 'tests'
    ]),
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
    ],
)
