import setuptools

setuptools.setup(
    name='aippa',
    version='0.1.0',
    author='Lorenzo Cardarelli',
    scripts=[],
    description='An awesome package that does something',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
    ]
)