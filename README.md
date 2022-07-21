
[![Build Status](https://dev.azure.com/karlritchie1/LIBiFBTSVM/_apis/build/status/kritchie.LIBiFBTSVM?branchName=master)](https://dev.azure.com/karlritchie1/LIBiFBTSVM/_build/latest?definitionId=1&branchName=master)

# LIBiFBTSVM

A Python library for an incremental and decremental fuzzy bounded twin support vector machine.

## Description

This library contains the implementation of the increment and decremental fuzzy bounded twin support vector machine [1].

## Usage

### Installation

The library is currently under active development, therefore, to install the development library you can
run the following command:

```bash
pip install git+https://github.com/kritchie/LIBiFBTSVM.git
```

Or, alternatively, you can clone the project and navigate to its root folder, then run the following command:

```bash
python setup.py install
```

### Examples

The `examples/` directory contains a few examples on how to use this library to perform classification tasks.

## Development

### Pre-requisites

This packages uses [pip-tools](https://github.com/jazzband/pip-tools) to manage its dependencies.
Development dependencies are located within the `requirements-dev.txt` file.

When adding dependencies to either `requirements.in` or `requirements-dev.txt`, make sure to call :

```
make compile-reqs
```

### Code structure

TODO

### Testing

The `tox` package is used to automate testing. The tox config is located in `tox.ini` and to run the testing pipeline, simply run the following :

```bash
tox
```

## References

1. de Mello, A. R., Stemmer, M. R., & Koerich, A. L. (2019). Incremental and Decremental Fuzzy Bounded Twin Support Vector Machine. arXiv preprint arXiv:1907.09613.
