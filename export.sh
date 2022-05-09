#!/bin/sh

rm -rf export

# loop over all notebooks and run nbconvert over them
for i in $(find . | grep -F .ipynb); do
  jupyter nbconvert --to markdown $i
done

mkdir export
mv ./*/*.md ./export
cp ./*/*.ipynb ./export