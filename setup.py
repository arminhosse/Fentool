"""Setup for building the package"""

import re
import subprocess

import setuptools

RESULT = subprocess.Popen(['/bin/bash', '-c', './git-tag.sh'], stdout=subprocess.PIPE)
VERSION_GIT = RESULT.communicate()[0]
TAG = VERSION_GIT.decode("utf-8")
VER = re.sub(r'\n', '', TAG)

setuptools.setup(
    name='Fentool',
    url='https://github.com/arminhosse/Fentool',
    version=VER,
    license='MIT',
    author='Armin Hosseini',
    author_email=['hosarmin@gmail.com'],
    package=setuptools.find_namespace_packages(include=['fentool.*'], exclude=["fentool.test.*"]),
    test_require=['unittest'],
    test_suite='unittest',
    desctiption='Feature Engineering Tool',
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent"
    ],
    install_requires=[
        'pandas',
        'scikit-learn',
        'unnitest'
    ]
)

