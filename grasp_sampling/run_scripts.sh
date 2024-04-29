#!/bin/bash

start=0    # initial start index
end=60     # initial end index (start + 60)

while (( start <= 500 )); do
    ./python.sh /media/juncheng/Disk4T1/Sim-Grasp/grasp_sampling/grasp_simulation.py --start_stage $start --end_stage $end || echo "Run with range $start-$((end-1)) encountered an error"
    start=$end
    end=$((end + 60))  # Increment end by 60
done
