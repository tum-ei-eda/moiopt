#!/bin/sh

set -e

python test_sched.py
python test_memplan.py
python test_fixreshapes.py
python test_pathdisc.py
python test_pathdiscfull.py
python test_moiopt.py
