#!usr/bin/env python
from setuptools import setup, find_packages
import sys
import os
packages = find_packages(exclude=['docs', 'notebooks', 'assets'])
install_requires = [
	'numpy>=1.17',
	'pandas',
	'networkx',
    'cvxpy',
	'matplotlib',
	'seaborn',
	'scikit-learn',
	'scikit-learn-extra',
	'stable-baselines3',
	'pyglet',
    'joblib',
    'gym',
]
with open("README.md", "r") as fh:
    long_description = fh.read()
setup(
    name='or-suite',
	version='0.0.1',
	description='OR-Suite: A set of environments for developing reinforcement learning agents for OR problems.',
	long_descrption=long_description,
	long_description_content_type='text/markdown',
	author='Christopher Archer, Siddhartha Banerjee, Shashank Pathak, Carrie Rucker, Sean Sinclair, Christina Yu',
	author_email = 'srs429@cornell.edu',
	license='MIT',
	url='https://github.com/seanrsinclair/ORSuite',
    packages=packages,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    zip_safe=False,
)
