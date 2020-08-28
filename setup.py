import os
import re

from setuptools import setup, find_packages


def requirements_from_file(filename='requirements.txt'):
    with open(os.path.join(os.path.dirname(__file__), filename)) as r:
        reqs = r.read().strip().split('\n')
    return [r for r in reqs if re.match(r'^\w+', r)]


setup(
    name="LIBiFBTSVM",
    license='BSD-3',
    version="0.0.1",
    url='https://github.com/kritchie/LIBiFBTSVM',
    packages=find_packages(),
    package_data={
        "": ["*.txt", "*.md"],
    },
    install_requires=requirements_from_file(),
    python_requires='>=3.6',
)
