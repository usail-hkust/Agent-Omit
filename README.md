# Agent-Omit: Training Efficient LLM Agents for Adaptive Thought and Observation Omission via Agentic Reinforcement Learning ([PDF](https://arxiv.org/pdf/2602.04284))

<p align="center">

![Testing Status](https://img.shields.io/badge/docs-in_progress-green)
![Testing Status](https://img.shields.io/badge/pypi_package-in_progress-green)
![License: CC BY 4.0](https://img.shields.io/badge/license-CC%20BY%204.0-blue)

</p>

<p align="center">

| **[Introduction](##introduction)** 
| **[Installation](##requirements)**
| **[Usage](##usage)**
| **[Citation](##citation)**

</p>

## Introduction
<div style="display: flex; justify-content: center;">
  <img src="https://github.com/usail-hkust/Agent-Omit/blob/yasNing/example/figure_method.png">
</div>

Agent-Omit is a framework that leverages **Agentic Reinforcement Learning** to teach Large Language Models (LLMs) to perform self-context management. By adaptively omitting redundant thoughts and observations, agents can achieve higher efficiency without compromising performance across diverse tasks.

This repository contains the implementation of Agent-Omit, built upon [AgentGym-RL](https://github.com/WooooDyy/AgentGym-RL) and [Verl](https://github.com/volcengine/verl).

## 🛠️ Installation

The installation consists of two parts:

1. **Agent Environments**: Hosting the specific task environments.
2. **Agentic RL Training**: Setting up the RL training environment.

### 1. Agent Environments Setup

We evaluate Agent-Omit on five distinct domains. Each environment is recommended to run in a separate conda environment to avoid conflicts.

#### 🌐 WebShop (Web Navigation)

*Navigating e-commerce websites for attribute extraction and purchasing.*

**Setup:**

```bash
cd AgentGym/agentenv-webshop
conda env create -n agentenv-webshop -f environment.yml
conda activate agentenv-webshop
bash ./setup.sh
```

**Launch Service:**

```
webshop --host 0.0.0.0 --port 36001
```

#### 🔍 DeepSearch (Information Search)

*Resolving knowledge-intensive queries via search engines.*

**Setup:**

```
cd AgentGym/agentenv-searchqa
conda env create -f environment.yml
conda activate agentenv-searchqa
pip install -e .
bash ./setup.sh
```

**Launch Service:**

```
searchqa --host 0.0.0.0 --port 36001
```

#### ⛏️ TextCraft (Digital Games)

*Minecraft-inspired crafting and long-horizon planning.*

**Setup:**

```
cd AgentGym/agentenv-textcraft
conda env create agentenv-textcraft python=3.9
conda activate agentenv-textcraft
pip install -e .
```

**Launch Service:**

```
textcraft --host 0.0.0.0 --port 36001
```

#### 🧪 SciWorld (Scientific Discovery)

*Complex reasoning in physical simulations.*

> **Note:** Requires Java 1.8+ installed on your system.

**Setup:**

```
conda create --name agentenv-sciworld python=3.8
conda activate agentenv-sciworld
pip install -e .
```

**Launch Service:**

Bash

```
sciworld --host 0.0.0.0 --port 36001
```

#### 👶 BabyAI (Embodied Control)

*Instruction following in partially observable grid-worlds.*

**Setup:**

```
conda create --name agentenv-babyai
conda activate agentenv-babyai
pip install -e .
```

**Launch Service:**

```
babyai --host 0.0.0.0 --port 36001
```

### 2. Agentic RL Training Framework Setup

This environment is used for the Agentic RL training loop. We recommend using **CUDA 12.4**, **PyTorch 2.4**, and **Python 3.10**.

```
echo "Preparing environment for agent-omit..."

# 1. Create Conda Environment
conda create -n agent-omit python==3.10 -y
conda activate agent-omit

# 2. Install PyTorch
pip3 install torch==2.4.0 --index-url [https://download.pytorch.org/whl/cu124

# 3. Install Flash Attention
FLASH_ATTENTION_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

FLASH_ATTENTION_NAME="flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

wget -q $FLASH_ATTENTION_URL -O $FLASH_ATTENTION_NAME
pip3 install $FLASH_ATTENTION_NAME
rm -f $FLASH_ATTENTION_NAME

# 4. Install agent-omit Core
cd AgentOmit-RL
pip3 install -e .

# 5. Install AgentEnv and Transformers
echo "Preparing environment for agentenv..."
cd ../AgentGym/agentenv
pip3 install -e .
pip3 install transformers==4.51.3
```

------

## 🚀 Usage

Before training or evaluation, ensure the target **Agent Environment server** is running (see *Agent Environments Setup* above).

### RL Training (WebShop Example)

To train the agent using Valiana GRPO on WebShop:

```
cd AgentOmit/example/AgentOmit-RL
bash ./web_train.sh
```

### Evaluation (WebShop Example)

To evaluate the trained checkpoint on WebShop:

```
cd AgentOmit/example/AgentOmit-Eval
bash ./webshop_eval.sh /your_model_dir/Qwen3-8B /your_log_dir/webshop_Qwen3-8B test
```

## 

## 📚 References

If you use Agent-Omit or the environments mentioned above, please cite the following works:

```
@article{ning2026agent,
  title={Agent-Omit: Training Efficient LLM Agents for Adaptive Thought and Observation Omission via Agentic Reinforcement Learning},
  author={Ning, Yansong and Fang, Jun and Tan, Naiqiang and Liu, Hao},
  journal={arXiv preprint arXiv:2602.04284},
  year={2026}
}

@article{ning2025not,
  title={Not all thoughts are generated equal: Efficient llm reasoning via multi-turn reinforcement learning},
  author={Ning, Yansong and Li, Wei and Fang, Jun and Tan, Naiqiang and Liu, Hao},
  journal={arXiv preprint arXiv:2505.11827},
  year={2025}
}
```

## 
