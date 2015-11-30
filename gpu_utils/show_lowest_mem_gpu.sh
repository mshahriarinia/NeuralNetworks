#!/bin/bash

# Copyright 2015 Mitsubishi Electric Research Labs (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# a script to show the lowest memory GPU id (based on nvidia-smi).

if [ -z `which nvidia-smi` ]; then
    echo "Error: nvidia-smi (cuda?) is not installed. Use CPU."  1>&2
    exit 1
fi

for gpu_id in `nvidia-smi -L | awk '{print $2}' | sed -e "s/://"`; do
    echo -n $gpu_id
    nvidia-smi -q -i $gpu_id -d MEMORY | grep Used | head -n 1
done | sort -n -k 4 | head -n 1 | awk '{print $1}'
