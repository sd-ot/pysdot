all: comp

comp:
	test -e ext/sdot || git clone git@github.com:sd-ot/sdot.git ext/sdot
	python3 setup.py build
	python3 setup.py install --user 

	
test:
	python -m unittest discover tests

.PHONY: test comp
