#!/bin/sh
directory=./data/test

mkdir -p $directory

wget -P $directory 'http://cv.snu.ac.kr/research/EDSR/benchmark.tar'

for file in $directory/*.tar
do
    tar -xvf $file -C $directory && rm $file 
done
