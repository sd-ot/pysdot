all: install

comp:
	test -e ext/sdot || git clone https://github.com/sd-ot/sdot.git ext/sdot
	rm -rf build/ 2> /dev/null
	python setup.py build

install: comp
	python setup.py install --user 
	
test:
	python -m unittest discover tests

pip:
	python setup.py sdist
	twine upload --verbose dist/*

.PHONY: test comp
