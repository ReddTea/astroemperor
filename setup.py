#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

#import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="astroemperor",
    version="0.9",
    author="ReddTea",
    author_email="redd@tea.com",
    description="PTMCMC sampler for exoplanet search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    #url="https://github.com/pypa/sampleproject",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    #ext_modules=extensions,
    #data_files=[(path1, [path1+'/exo_list.csv', path1+'/ss_list.csv'])],
    include_package_data=True,
    #setup_requires=['numpy', 'cython'],
    install_requires=['numpy', 'matplotlib>=3.5.1',
                      'ipywidgets', 'tabulate', 'termcolor', 'reddutils', 'kepler.py',
                      'emcee', 'arviz', 'corner', 'reddcolors',
                      'reddemcee'],
    python_requires=">=3.6",
)

