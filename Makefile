all: install

comp:
	test -e ext/sdot || git clone https://github.com/sd-ot/sdot.git ext/sdot
	rm -rf build/ 2> /dev/null
	python3 setup.py build

install: comp
	python3 setup.py install --user 
	
test:
	python -m unittest discover tests

.PHONY: test comp
