#!/usr/bin/env python3

from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution
import setuptools.command.build_py
import subprocess
import sys
import os

extra_compile_args = []
if 'darwin' in sys.platform:
    extra_compile_args.append("-std=c++14")
    extra_compile_args.append("-stdlib=libc++")
    extra_compile_args.append("-Wno-missing-braces")
    extra_compile_args.append("-march=native")
    extra_compile_args.append("-ffast-math")
if 'linux' in sys.platform:
    extra_compile_args.append("-march=native")
    extra_compile_args.append("-ffast-math")

ext_modules = []
for TF in ["double"]:
    for dim in [2, 3]:
        name = 'pybind_sdot_{}d_{}'.format(dim, TF)
        ext_modules.append(Extension(
            name,
            sources=['pysdot/cpp/pybind_sdot.cpp'],
            include_dirs=['ext','ext/pybind11/include/'],
            define_macros=[
                ('PD_MODULE_NAME', name),
                ('PD_TYPE', TF),
                ('PD_DIM', str(dim))
            ],
            language='c++',
            extra_compile_args=extra_compile_args,
        ))

class BuildPyCommand(setuptools.command.build_py.build_py):
    """Custom build command."""

    def run(self):
        if not os.path.isdir('./ext/sdot'):
            subprocess.check_call(['git', 'clone', 'https://github.com/sd-ot/sdot.git', 'ext/sdot'])
        if not os.path.isdir('./ext/eigen3'):
            subprocess.check_call(['git', 'clone', 'https://github.com/eigenteam/eigen-git-mirror.git', 'ext/eigen3'])
        if not os.path.isdir('./ext/pybind11'):
            subprocess.check_call(['git', 'clone', 'https://github.com/pybind/pybind11.git', 'ext/pybind11'])
        setuptools.command.build_py.build_py.run(self)

setup(
    name='pysdot',
    version='0.1.post11',
    packages=find_packages(exclude=[
        'hugo', 'ext', 'build', 'dist',
        'examples', 'results', 'tests'
    ]),
    cmdclass={
        'build_py': BuildPyCommand,
    },
    include_package_data=True,
    ext_modules=ext_modules,
    install_requires=[
        "numpy",
    ],
    author="Hugo Leclerc",
    author_email="hugal.leclerc@gmail.com",
    description="Semi-discrete operationnal transport",
    long_description="""
        Semi-discrete operationnal transport for the masses...
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/sd-ot/pysdot",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

