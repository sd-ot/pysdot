mkdir -p ext
git clone https://github.com/sd-ot/sdot.git ext/sdot
"%PYTHON%" setup.py build
"%PYTHON%" setup.py install
