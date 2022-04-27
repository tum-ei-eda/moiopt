#!/bin/sh

set -e

python test_sched.py
python test_memplan.py
python test_moiopt.py
