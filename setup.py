#!/usr/bin/env python3

from setuptools import setup, find_packages, Extension
from setuptools.dist import Distribution
import setuptools.command.build_py
import subprocess
import sys
import os

extra_compile_args = []
if 'darwin' in sys.platform:
    extra_compile_args.append("-std=c++17")
    extra_compile_args.append("-stdlib=libc++")
    extra_compile_args.append("-Wno-missing-braces")
    # extra_compile_args.append("-march=nocona")
    extra_compile_args.append("-ffast-math")
if 'linux' in sys.platform:
    # extra_compile_args.append("-march=nocona")
    extra_compile_args.append("-ffast-math")
    extra_compile_args.append("-std=c++17")

ext_modules = []

include_dirs = [ 'boost_1_87_0', 'ext/eigen3', 'ext/pybind11/include', '/usr/share/miniconda/envs/test/include', '$PREFIX/include', '$CONDA_PREFIX/include' ]
for ev in [ ( "CONDA_PREFIX", "/include" ), ( "PREFIX", "/include" ), ( "BUILD_PREFIX", "/include" ), ( "LIBRARY_INC", "" ) ]:
    try:
        include_dirs.append( os.environ.get( ev[ 0 ] ) + ev[ 1 ] )
    except:
        pass

# Arfd
for ext in ["Arfd"]:
    ext_modules.append(Extension(
        "pybind_sdot_" + ext,
        sources=['pysdot/cpp/pybind_sdot_' + ext + '.cpp'],
        include_dirs=include_dirs,
        language='c++',
        extra_compile_args=extra_compile_args,
    ))

# PowerDiagram
for TF in ["double"]:
    for dim in [2, 3]: # 
        name = 'pybind_sdot_{}d_{}'.format(dim, TF)
        ext_modules.append(Extension(
            name,
            sources=['pysdot/cpp/pybind_sdot.cpp'],
            include_dirs=include_dirs,
            define_macros=[
                # ('PD_WANT_STAT', ""),
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
            subprocess.run(['git', 'clone', 'https://github.com/sd-ot/sdot.git', 'ext/sdot'], check=True)
        if not os.path.isdir('./ext/eigen3'):
            subprocess.run(['git', 'clone', 'https://github.com/eigenteam/eigen-git-mirror.git', 'ext/eigen3'], check=True)
        if not os.path.isdir('./ext/pybind11'):
            subprocess.run(['git', 'clone', 'https://github.com/pybind/pybind11.git', 'ext/pybind11'], check=True)
        if not os.path.isdir('./boost_1_87_0'):
            # subprocess.run(['git', 'clone', 'https://github.com/boostorg/dynamic_bitset.git', 'ext/dynamic_bitset'], check=True)
            subprocess.run('curl -L -o boost_1_87_0.tar.gz https://archives.boost.io/release/1.87.0/source/boost_1_87_0.tar.gz', check=True, shell=True)
            subprocess.run(['tar', 'xzf', 'boost_1_87_0.tar.gz'], check=True)
        setuptools.command.build_py.build_py.run(self)

setup(
    name='pysdot',
    version='0.2.35',
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

