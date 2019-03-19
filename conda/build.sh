#!/bin/bash

# if [[ $(uname) == Darwin ]]; then
#     EXTRA_ARGS="--without-threads"
# fi

mkdir -p ext
git clone https://github.com/sd-ot/sdot.git ext/sdot
python setup.py build
python setup.py install
