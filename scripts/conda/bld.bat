mkdir -p ext

git clone https://github.com/sd-ot/sdot.git ext/sdot
git clone https://github.com/eigenteam/eigen-git-mirror.git ext/eigen3
"%PYTHON%" setup.py build
"%PYTHON%" setup.py install

