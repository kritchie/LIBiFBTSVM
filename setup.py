from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="LIBiFBTSVM",
    license='BSD-3',
    version="0.0.1",
    url='https://github.com/kritchie/LIBiFBTSVM',

    ext_modules=cythonize('./libifbtsvm/functions/ctrain_model.pyx'),
    packages=find_packages(),
    package_data={
        "": ["*.txt", "*.md"],
    },

    python_requires='>=3.6',
)
