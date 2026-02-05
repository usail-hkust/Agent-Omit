#!/bin/bash

__conda_setup="$('/nfs/ofs-llm-ssd/user/oliverleo/miniconda3' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/nfs/ofs-llm-ssd/user/oliverleo/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/nfs/ofs-llm-ssd/user/oliverleo/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/nfs/ofs-llm-ssd/user/oliverleo/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate /nfs/ofs-llm-ssd/user/ningyansong/envs/agentenv-webshop
port=$1
# launch the server
cd /your_path_/agentenv-webshop
webshop --host 0.0.0.0 --port "$port" > "webshop_${port}.log" 2>&1 &