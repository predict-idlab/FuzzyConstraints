#!/bin/bash

for x in TransE DistMult
do
for y in reference clustering lcwa strict standard hybrid
do
for z in AIFB MUTAG
do
python -u tests.py $x $y $z 0 0 0 0 1
python -u tests.py $x $y $z 0 0 0 0 5
python -u tests.py $x $y $z 0 1 1 0 1
python -u tests.py $x $y $z 0 1 1 0 5
done
done
done