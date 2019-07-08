#!/usr/bin/env python

from distutils.core import setup

install_requires=[
    'tensorflow>=1.12',
    'keras',
    'numpy',
    'scipy',
    'matplotlib',
    'astropy'
    'root-numpy'
]

setup(name='pe_extractor',
      version='1.0',
      description='extract photo-electrons from waveforms.',
      author='Yves Renier',
      url='https://github.com/cta-sst-1m/pe_extractor',
      packages=['pe_extractor'],
     )

