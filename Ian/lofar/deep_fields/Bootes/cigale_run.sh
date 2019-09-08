#!/bin/bash

source activate cigalon
cd data/cigale_results/sample

folder_list=$(ls)

for folder in $folder_list
do
    echo $folder
    if [[ "$folder" =~ "_z_" ]]
    then
        cd $folder
        #pcigale check
        #pcigale run
        pcigale-plots sed
        cd ..
    fi
done