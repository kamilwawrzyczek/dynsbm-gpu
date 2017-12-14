#!/bin/bash
gpu=true
iterations=10
init_mehods=("random" "k-means")
distributions=("uniform" "normal" "power")
start_N=0
end_N=400

set -e
echo "File|Iteration|Device|Initialization method|N|Graph generate time|Execution time|Total elapsed time|Iterations|Iterations internal|Memory CPU|Memory GPU|Log likelihood|Log likelihood iterations"
for file in `ls -v DataSet/*`
do
    N=`echo ${file} | cut -d '_' -f 2`
    distribution=`echo ${file} | cut -d '_' -f 5 | cut -d '.' -f 1`
	if [[ ${N} -le ${end_N} && ${N} -ge ${start_N} && " ${distributions[@]} " =~ " ${distribution} " ]]
	then
	    for init_method in "${init_mehods[@]}"
	    do
		    for iteration in $(seq 1 $iterations)
		    do
		        python3 ./dynsbm-gpu/main_run_set.py $file $gpu $init_method $iteration
		    done	
	    done
    fi
done
