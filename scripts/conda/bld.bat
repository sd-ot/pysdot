mkdir -p ext

git clone https://github.com/sd-ot/sdot.git ext/sdot
VS140COMNTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\Common7\Tools "%PYTHON%" setup.py build
"%PYTHON%" setup.py install
