from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import glob

__version__ = "0.0.1"

setup(
    name='dynamic_plots',
    version=__version__,
    author='Carlo Emilio Montanari, Federico Panichi',
    author_email='carlidel95@gmail.com',
    url='https://github.com/carlidel/dynamic_plots',
    description='Some interactive plots in matplotlib for analyzing dynamic indicators',
    long_description='',
    packages=["dynamic_plots"],
    install_requires=['numba', 'numpy', 'scipy', 'matplotlib'],
    setup_requires=['numba', 'numpy', 'scipy', 'matplotlib'],
    license='MIT',
)
