import setuptools

with open('README.md','r') as file:
    long_description = file.read()

setuptools.setup(
name="pygrc",
version="0.3.0",
description= "A package to read SPARC data for Galactic Rotation Curves",
author="Aman Desai",
author_email="amanmukeshdesai@gmail.com",
maintainer="Aman Desai",
maintainer_email="amanmukeshdesai@gmail.com",
url = "https://github.com/amanmdesai/pygrc",
long_description=long_description,
packages=["pygrc"],
install_requires=["numpy","pandas","iminuit","matplotlib","seaborn"],
long_description_content_type="text/markdown",
classifiers=["License :: OSI Approved :: MIT License",
             "Programming Language :: Python :: 3",
             "Programming Language :: Python :: 3.10",
             "Programming Language :: Python :: 3.11",
             "Programming Language :: Python :: 3.12",
             "Topic :: Scientific/Engineering :: Physics",
             "Operating System :: Unix",
             "Operating System :: Microsoft :: Windows",
             "Operating System :: MacOS"]
)
