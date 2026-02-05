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
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

task_name="babyai"


conda activate agent-omit
export WANDB_BASE_URL=https://api.bandw.top


FREE_PORT=$(python3 /your_path/example/AgentOmit-RL/find_free_port.py)

# 输出分配的端口号（可选）
echo "Assigned port: ${FREE_PORT}"
# launch the textcraft server
bash /your_path/AgentOmit-Gym/agentenv-babyai/start.sh ${FREE_PORT}

env_server_url="http://0.0.0.0:${FREE_PORT}"


export WANDB_API_KEY="your_wandb_key"

agent_model_path="/your_path/Qwen3-8B"

kl_coef=0.001
policy_learning_rate=5e-7
rollout_sample_num=8
train_batch_size=8
ppo_mini_batch_size=8
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=1

total_epoches=1

model_save_dir="/your_path/agent-gym"
exp_name="babyai_efficient_Qwen3-8B"
model_save_path=${model_save_dir}/${exp_name}

mkdir -p ${model_save_path}

cd /your_path/AgentOmit/AgentOmit-RL
HYDRA_FULL_ERROR=1 WANDB_MODE=online python3 -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=10 \
    data.train_file=AgentTrain/${task_name}_train.json \
    data.train_batch_size=${train_batch_size} \
    data.val_files=AgentTrain/${task_name}_val.json \
    data.val_batch_size=96 \
    data.max_prompt_length=28000 \
    data.max_response_length=2048 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=32000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=320000 \
    actor_rollout_ref.rollout.max_tokens=2048 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=xxx \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    trainer.val_before_train=True \
    trainer.total_epochs=${total_epoches} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \