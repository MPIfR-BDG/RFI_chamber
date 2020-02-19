from setuptools import setup, find_packages

setup(name='rfi_chamber',
      version='1.0',
      description='Python script for running spectrometer captures',
      url='https://github.com/MPIfR-BDG/RFI_chamber',
      author='Ewan Barr',
      author_email='ebarr@mpifr-bonn.mpg.de',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'pyvisa',
          'pyvisa-py',
          'pyyaml',
          'coloredlogs',
          'astropy'
      ],
      dependency_links=[
      ],
      zip_safe=False)
