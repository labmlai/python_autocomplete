#!/bin/bash

mkdir data/source
cp data/download/*.zip data/source
cd data/source
unzip -o \*.zip
rm *.zip
cd .. & cd ..
