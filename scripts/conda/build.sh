#!/bin/bash

# if [[ $(uname) == Darwin ]]; then
#     EXTRA_ARGS="--without-threads"
# fi

mkdir -p ext
git clone https://github.com/sd-ot/sdot.git ext/sdot
git clone https://github.com/eigenteam/eigen-git-mirror.git ext/eigen3
python setup.py build
python setup.py install
