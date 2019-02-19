.PHONY: .FORCE

build: .FORCE
	python setup.py sdist bdist_wheel

upload: .FORCE
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

install-dev: .FORCE
	#a) clone and install
	echo "python setup.py develop"
	#b) install via git+
	echo "python -m pip install -e git+https://github.com/iurisegtovich/pytherm-lib.git@master#egg=pytherm_lib_iurisegtovich"
