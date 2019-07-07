#!/usr/bin/env python

from distutils.core import setup

install_requires=[
    'tensorflow>=1.12',
    'numpy',
    'scipy',
    'matplotlib',
]

setup(name='pe-extractor',
      version='1.0',
      description='extract photo-electrons from waveforms.',
      author='Yves Renier',
      url='https://github.com/cta-sst-1m/pe-extractor',
      packages=['pe-extractor'],
     )

