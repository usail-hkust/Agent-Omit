#!/bin/bash


__conda_setup="$('/your_conda_str/miniconda3' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/your_conda_str/etc/profile.d/conda.sh" ]; then
        . "/your_conda_str/etc/profile.d/conda.sh"
    else
        export PATH="/your_conda_str/bin:$PATH"
    fi
fi
unset __conda_setup

set -x
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

task_name="textcraft"

conda activate agent-omit


# 调用 Python 脚本并获取未被占用的端口号
FREE_PORT=$(python3 /your_path/example/AgentOmit-RL/find_free_port.py)
# 输出分配的端口号（可选）
echo "Assigned port: ${FREE_PORT}"

bash /your_path/AgentOmit-Gym/agentenv-textcraft/start.sh ${FREE_PORT}

env_server_url="http://0.0.0.0:${FREE_PORT}"

sample_num=1
max_rounds=12

model_path=$1
executer_logs=$2
gpus_nums=1
sample_category=$3

cd /your_path/AgentOmit/AgentOmit-RL
HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation  \
    data.path=AgentEval/${task_name} \
    data.max_prompt_length=28000 \
    data.max_response_length=2048 \
    data.n_samples=${sample_num} \
    data.batch_size=${gpus_nums} \
    agentgym.task_name=${task_name} \
    +agentgym.sample_category=${sample_category} \
    agentgym.env_addr=${env_server_url} \
    agentgym.max_rounds=${max_rounds} \
    agentgym.timeout=500 \
    model.path=${model_path} \
    rollout.gpu_memory_utilization=0.95 \
    rollout.temperature=0.3 \
    rollout.max_model_len=32000 \
    rollout.max_num_batched_tokens=320000 \
    rollout.max_tokens=2048 \
    rollout.tensor_model_parallel_size=${gpus_nums} \
    rollout.rollout_log_dir=${executer_logs} \
    trainer.n_gpus_per_node=${gpus_nums} \