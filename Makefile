all: install

comp:
	test -e ext/sdot || git clone https://github.com/sd-ot/sdot.git ext/sdot
	rm -rf build/ dist/ pysdot.egg-info/ 2> /dev/null
	python setup.py build

install_user: comp
	python setup.py install --user 
	
install: comp
	python setup.py install
	
test:
	python -m unittest discover tests

pip:
	rm -rf dist/*
	python setup.py sdist
	twine upload --verbose dist/*

.PHONY: test comp
