<div align="center">
<img width="478" height="150" src="docs/images/kaggen-banner-transparent.png">

**Flexible template code for deep learning pipelines**

[![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- [![PyPI Status](https://badge.fury.io/py/kaggen.svg)](https://badge.fury.io/py/kaggen) [![Downloads](https://pepy.tech/badge/kaggen)](https://pepy.tech/project/kaggen) ![Build Status](https://github.com/jeffrey_wardman/kaggen/workflows/Build/badge.svg) [![codecov](https://codecov.io/gh/jeffrey_wardman/kaggen/branch/master/graph/badge.svg)](https://codecov.io/gh/jeffrey_wardman/kaggen) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->

</div>

---

Kaggen is a light-weight, flexible Python library for quick prototyping for deep learning pipelines. The library originated from observations that notebooks in Kaggle competitions are typically based on the formatting of the first popular notebooks for training and inferencing pipelines. It's not rare for these to consist of non-production level code, (e.g. variables whose meaning is unclear).

The idea of this library is for it to be forked and used as the basis for these pipelines, allowing users to remain flexible in their approaches whilst maintaining good software engineering principles. Basic changes can be done simply via the YAML config files, and various external resources can be included. More detailed changes can be done by forking and tweaking the code.

# Features

Kaggen allows preprocessing, training and inference pipelines to be generated (currently limited to classification for image-based problems).

### Coming Soon

- Convert python script/s to Jupyter notebook
- Custom models with shape determined from YAML file.
  - PyTorch model -> YAML file
  - External resources for layers
- Segmentation and other problems out-of-the-box
- Submit directly to Kaggle

# Installation

Install kaggen from PyPI with pip:

```
$ pip install kaggen
```

Or fork the repository and build on top of the library with

```
$ git clone git@github.com:JeffreyWardman/kaggen.git
$ cd kaggen
$ poetry install kaggen
```

**Notes:**

- Requires Python 3.7+
- Tested on Python 3.8

# How to Use

Fork/clone and pip install package by navigating to directory and typing:

```
$ pip3 install -e .
```

Run/debug like any other module. For example:

```
$ python3.7 -m kaggen.train
```

To use a custom function, set the module name to the file containing the function, e.g. kaggen.train.optimizers and set the name to be the function name, e.g. SGD. For example:

```
optimizer:
  module: kaggen.train.optimizers
  name: SGD
```

To use a different/custom model, create a new class in the models file.

## How to use in offline Kaggle competition

- Build package wheel
- Upload to Kaggle
- Install from wheel

# Documentation

Build documents locally with:

```
$ cd docs && make html
```

### Tips

Create files where needed and use the below to import a functions from within the same file, or from other modules

```
# Same file
eval('function_name')(**input_args)

# Other modules
geattr(importlib.import_module('module_name'), 'function_name')(**input_args)
```

<!-- # Citing

If you are using kaggen for your research project or competition entry, we ask that you please cite it using one of the following bibtex citations:

    @inproceedings {kaggen,
        author = {Jeffrey Wardman},
        title = "",
        publisher = "",
        year = 2020
    } -->

# Issues and Contributions

See the [contribution guide](https://github.com/JeffreyWardman/kaggen/wiki/Contributing) for detailed information. Please see the issues page before contacting the author.

# Contributors

- Jeffrey Wardman -- <http://github.com/jeffreywardman>

# License

Kaggen is under an [MIT License](https://opensource.org/licenses/MIT)
