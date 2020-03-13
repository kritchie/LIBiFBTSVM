
[![Build Status](https://dev.azure.com/karlritchie1/LIBiFBTSVM/_apis/build/status/kritchie.LIBiFBTSVM?branchName=master)](https://dev.azure.com/karlritchie1/LIBiFBTSVM/_build/latest?definitionId=1&branchName=master)

# LIBiFBTSVM

A Python library for an incremental and decremental fuzzy bounded twin support vector machine.

## Description

This library contains the implementation of the increment and decremental fuzzy bounded twin support vector machine [1].

## Installation

In order to be able to use this package, the first step is to clone this repository:

```bash
git clone https://github.com/kritchie/LIBiFBTSVM.git
```

Navigate to the cloned repository and call the following command (it is recommded to do so in a virtual environment):

```bash
python setup.py --install
```

## Usage

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
