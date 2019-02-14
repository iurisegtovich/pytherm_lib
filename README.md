# repo

This is a simple repo for the project


1. FOLLOWING TUTORIAL AT https://packaging.python.org/tutorials/packaging-projects/

2. BUILD PACKAGE: python setup.py sdist bdist_wheel

3. UPLOAD TEST: python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

4. INSTALL: python -m pip install --index-url https://test.pypi.org/simple/ python-luri-iurisegtovich

5. UPLOAD REAL: python -m twine upload dist/*

6. INSTALL REAL: python -m pip install python-luri-iurisegtovich

7. TEST: import python_luri

8. a. UPLOAD NEW VERSION: Open setup.py and change the version number
8. b. DELETE OLD VERSION INSIDE DIST* (BETTER TO DELETE ALL DIST RELATED FILES)
8. c. REINSTALL (IGNORING PREVIOUS VERSIONS) python -m pip install -I --index-url https://test.pypi.org/simple/ python-luri-iurisegtovich

obs: O SETUP.PY É O ARQUIVO DE RECEITA PARA GERAR O RELEASE,
ELE PEGA O README DAQUI POIS ISSO ESTÁ INSTRUÍDO
ELE PEGA OS ARQUIVOS DE DENTRO DA PASTA DO PACKAGE (setuptools.find_packages())
ELE PARECE QUE NÃO PEGOU O LICENSE (COLOQUEI UMA LINHA EXTRA NO SETUP.PY PARA PEGAR)

acho que posso iniciar um git aqui e colocar esses comandos em um makefile
e rodar um git clean para apagar os arquivos de build

9. INSTALL FROM ZIP: python -m pip install -I dist/python_luri_iurisegtovich-0.0.3-py3-none-any.whl

10. "INSTALL DEVELOPMENT VERSION": 
IN ANOTHER PROJECT DIRECTORY
pip install --e git+http://repo/my_project.git#egg=SomeProject
[https://pip.pypa.io/en/stable/reference/pip_install/]
python -m pip install -e git+https://github.com/iurisegtovich/python_luri.git@master#egg=python_luri_iurisegtovich
> install in a "src" dir under the dir from where pip was ran.

> any changes in those src files are reflected on usage of the lib
> the lib can be imported from anywhere as if it was installed in site-packages (actually because the created src dir is added to the sys.path variable)

> check with: "python -m pip list"
> uninstall with: python -m pip uninstall python_luri_iurisegtovich





