#!/bin/bash

cd "$(dirname "$0")"

data=$1

echo $data

./rerx2.sh $data
./j48graft2.sh $data
