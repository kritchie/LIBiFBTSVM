from setuptools import setup, find_packages

setup(
    name="LIBiFBTSVM",
    license='BSD-3',
    version="0.0.1",
    url='https://github.com/kritchie/LIBiFBTSVM',

    packages=find_packages(),
    package_data={
        "": ["*.txt", "*.md"],
    },

    python_requires='>=3.6',
)
