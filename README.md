# Pytunnel package


## Introduction

This repository contains scripts to run virtual corridors using Python 3.
2023/05/17
This repository is for use in behavioral box.
Cloned from Shohei Furutachi's project.


## Installation

To retrieve the code, clone the repository using git:
```
git clone https://bitbucket.org/lasermouse/pytunnel 
```

To install dependencies, I recommend that you first create a virtual
environment (with python or conda) and install dependencies inside using pip.

Using python3 (from a terminal), recommended for Linux users:
```bash
python3.5 -m venv pytunnel_venv
source pytunnel_venv/bin/activate
pip install -r requirements.txt
```

Using conda (from Anaconda Prompt), recommended for Windows users:
```bash
conda create -n pytunnel_venv python=3.5
conda activate pytunnel_venv
pip install -r requirements.txt
```


## Usage

To start a virtual corridor you need to:
- open a terminal
- activate the virtual environment
- run a tunnel script with the corresponding tunnel configuration file.

For example, to run one the example file for the basic tunnel:
```bash
conda activate pytunnel_venv
python src/pytunnel/flip_tunnel.py examples/flip_tunnel_example.yaml
```

To know more about the options of a tunnel script, use the `--help`/`-h` option, e.g.
```bash
python src/pytunnel/flip_tunnel.py --help
```


## TODOs

- `TODO` document tunnel configuration
- `TODO` document texture creation
- `TODO` document the code
- `TODO` give credit to `Aris`
- `TODO` add a licence
