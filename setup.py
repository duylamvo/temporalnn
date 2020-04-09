from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = find_packages(where=".", exclude=("tests*",))
package_dir = {'': 'temporalnn'}
setup(
    name="temporalnn",
    version="0.0.2",
    author="Duy Lam Vo",
    author_email="dungthuapps@gmail.com",
    description="A small utils for temporal neural network",
    long_description=long_description,
    url="https://github.com/dungthuapps/temporalnn",
    # package_dir=package_dir,
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'

)
