
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

(Make sure to have `poetry` installed in your environment)

```bash
make install
```

### Examples

The `examples/` directory contains a few examples on how to use this library to perform classification tasks.

## Development

### Pre-requisites

This packages uses [poetry](https://python-poetry.org/) to manage its dependencies.  You can install like running the following command:

```
pip install poetry
```

### Testing

To run the testing pipeline, simply run the following :

```bash
make test
```

## References

1. de Mello, A. R., Stemmer, M. R., & Koerich, A. L. (2019). Incremental and Decremental Fuzzy Bounded Twin Support Vector Machine. arXiv preprint arXiv:1907.09613.
