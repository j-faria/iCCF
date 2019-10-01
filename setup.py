""" iCCF """

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='iCCF',
    version='0.3.5',
    description='Analysis of CCF profiles and activity indicators',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/j-faria/iCCF',
    author='João Faria',
    author_email='joao.faria@astro.up.pt',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
      entry_points={
        'console_scripts': [
            'iccf-fits-to-rdb = iCCF.scripts:fits_to_rdb',
            'iccf-make-ccf = iCCF.scripts:make_CCF',
            ]
        },
    packages=find_packages(),
    install_requires=['numpy', 'cached_property'],
)
