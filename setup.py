import os
import sys
import subprocess
from setuptools import setup, find_packages
from distutils.cmd import Command

setup(
    name='eva',
    version='0.1.0',
    author='Tommy Chang',
    author_email='tomy.1516@gmail.com',
    description='EVA: A tool for submitting jobs to HPC clusters.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    package_dir={
        "": "src"
    },
    include_package_data=True,
    install_requires=[
        'numpy',
        'orjson',
        'multiset',
        'boto3',
        'grpcio',
        'grpcio-tools',
        'docker',
        'torch',
        'build'
    ],
    entry_points={
        'console_scripts': [
            'eva-submit=eva_submit:main',
            'eva-worker=eva_worker:main',
            'eva-master=eva_master:main',
        ],
    },
    classifiers=[
        # Classifiers help users find your project by categorizing it.
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6'
)
