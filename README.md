# MOIOPT

Memory-Optimizing, Inter-Operator Tiling (MOIOPT) is a method to reduce peak memory usage for TinyML applications. It applies a set of inter-operator tiling techniques that work across a wide range of layer types to split buffers that cause memory peaks.

## Requirements

Firstly, [install Apache TVM](https://tvm.apache.org/docs/install/index.html).

The following system packages are required, given as Debian/Ubuntu example:

    sudo apt install python3-virtualenv

Next, install the python dependencies, for example with virtualenv:

    cd moiopt
    virtualenv -p python3 venv
    . venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

## Usage

Run the `MOIOPTPass` in `src/moiopt.py` after all built-in Relay optimization passes. Unfortunately, TVM currently does not provide a hooking point at this location, but an example of a possible modification can be seen here: https://discuss.tvm.apache.org/t/insert-relay-pass-after-optimizations/12664

## Cite this work

This work has been described in the following paper: https://arxiv.org/abs/2303.17878

```
@article{stahl2023fused,
  title={Fused Depthwise Tiling for Memory Optimization in TinyML Deep Neural Network Inference},
  author={Stahl, Rafael and Mueller-Gritschneder, Daniel and Schlichtmann, Ulf},
  booktitle={arXiv preprint arXiv:2303.17878},
  year={2023}
}
```
