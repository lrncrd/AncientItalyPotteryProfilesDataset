import setuptools

setuptools.setup(
    name='aippa',
    version='0.2.0',
    author='Lorenzo Cardarelli',
    author_email='lorenzo.cardarelli@uni-goettingen.de',
    scripts=[],
    description='An awesome package that does something',
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.12",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "openpyxl",
        "scikit-learn",
        "scikit-image",
        "matplotlib",
        "seaborn",
        "plotly",
        "bokeh",
        "umap-learn[plot]",
        "hdbscan",
        "pyefd",
        "tqdm",
        "pillow",
        "geopandas",
        "ipykernel",  # AIPPA is meant for Jupyter: plotly renders in the notebook
        "nbformat",
    ]
)
