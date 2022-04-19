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
