# Example Package

This is a simple example package. You can use
[Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)
to write your content.

1. FOLLOWING TUTORIAL AT https://packaging.python.org/tutorials/packaging-projects/

2. BUILD PACKAGE: python setup.py sdist bdist_wheel

3. UPLOAD TEST: python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

4. INSTALL: python -m pip install --index-url https://test.pypi.org/simple/ example-pkg-iurisegtovich

5. UPLOAD REAL: python -m twine upload dist/*

6. INSTALL REAL: python -m pip install example-pkg-iurisegtovich

7. TEST: import example_pkg

8. a. UPLOAD NEW VERSION: Open setup.py and change the version number
8. b. DELETE OLD VERSION INSIDE DIST* (BETTER TO DELETE ALL DIST RELATED FILES)
8. c. REINSTALL (IGNORING PREVIOUS VERSIONS) python -m pip install -I --index-url https://test.pypi.org/simple/ example-pkg-iurisegtovich

obs: O SETUP.PY É O ARQUIVO DE RECEITA PARA GERAR O RELEASE,
ELE PEGA O README DAQUI POIS ISSO ESTÁ INSTRUÍDO
ELE PEGA OS ARQUIVOS DE DENTRO DA PASTA DO PACKAGE (setuptools.find_packages())
ELE PARECE QUE NÃO PEGOU O LICENSE

acho que posso iniciar um git aqui e colocar esses comandos em um makefile
e rodar um git clean para apagar os arquivos de build
