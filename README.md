DataKnobs
=============================

## Description

Useful implementations of data structures and design patterns for knowledge bases and AI, or the knobs and levers for fine-tuning and leveraging your data.

This repo also serves as a template or sandbox for development, experimentation, and testing of general data structures, algorithms, and utilities for DS, AI, ML, and NLP.

Provides connectors for other popular text and data processing packages like:
  * numpy and pandas
  * nltk
  * wordnet
  * postgres
  * elasticsearch

## General project information

The purpose of this project is:

  * To provide dependable implementations of useful data structures.
  * To show examples of design patterns and ways to apply AI concepts.
  * To prototype tools for delivering a robust DS/AI/ML/NLP utilities library package.
  * To facilitate interactive development, demonstration, visualization, and testing of the library components via jupter notebooks and/or scripts.

## Installation and Usage

```bash/python
% pip install dataknobs
% python
>>> import dataknobs as dk
>>> ...
```


## Development

### Development machine prerequisites

The following minimum configuration should exist for development:

  * tox
  * pyenv
     * pyenv install 3.9
  * poetry

With optional:

  * docker
  * bash

By convention, a data directory can be leveraged for development that is mounted as a shared volumne in Docker as /data. This has the default of $HOME/data, but can be overridden with the DATADIR environment variable.


### Development quickstart guide

  * In a terminal, clone the repo and cd into the project directory.

#### Testing

  * Tests and Lint: "tox"
  * Just unit tests: "tox -e tests"
  * Just lint: "tox -e lint"

#### Using docker

  * Development:
```
% tox -e dev
# poetry shell
# python
```

  * Notebook:
    * execute "tox -e nb"
      * copy/paste url into browser

#### Using virtual environments

  * Development:
    * Manual: source ".project_vars", poetry install, poetry shell
    * Automated: execute "bin/start_dev.sh"  (requires "/bin/bash" on your machine)

  * Notebook:
    * execute "bin/start_notebook.sh"
      * copy/paste url into browser
