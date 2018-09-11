#!/bin/bash
mkdir -p models
for DB in REC-Street YT-All YT-Music YT-Clean
do
	wget http://www.svcl.ucsd.edu/~morgado/spatialaudiogen/models/${DB}.tar.gz
	tar xfvz ${DB}.tar.gz --directory models/
	rm xfvz ${DB}.tar.gz
done
