""" iCCF """

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the version from iCCF/version.py
version = {}
with open("iCCF/version.py") as fp:
    exec(fp.read(), version)
    __version__ = version['__version__']


setup(
    name='iCCF',
    version=__version__,
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
    package_data={
        '': ['example_data/*.npy', 'data/*'],
    },
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'matplotlib',
                      'astropy', 'cached_property', 'paramiko'],
)
