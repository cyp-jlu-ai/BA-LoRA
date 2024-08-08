# Bias-Aware Low-Rank Adaptation: Mitigating Catastrophic Inheritance of Large Language Models

## Introduction
 Large language models (LLMs) have achieved remarkable performance across a wide range of natural language processing (NLP) tasks. However, deploying LLMs on downstream tasks typically involves computationally expensive and memory-intensive fine-tuning. Parameter-efficient fine-tuning (PEFT) techniques have emerged as a promising solution to adapt LLMs with minimal computational overhead. Despite their advantages, LLMs still face the challenge of inheriting and amplifying biases involved in the pre-training data. In this work, we propose a novel PEFT method called Bias-Aware Low-Rank Adaptation (BA-LoRA), which incorporates three different regularization terms: (1) consistency regularizer, (2) diversity regularizer, and (3) singular vector decomposition regularizer, to address this issue. In particular, these regularizers aim to improve the generative models' consistency, diversity, and generalization capabilities in fine-tuning. We conduct extensive experiments on a diverse set of tasks, including natural language understanding (NLU) and natural language generation (NLG), using mainstream LLMs such as LLaMA, Mistral, and Gemma. The empirical findings demonstrate that BA-LoRA outperforms Low-Rank Adaptation (LoRA) and its state-of-the-art variants. Furthermore, our approach is demonstrated to effectively mitigate biases inherited from the pre-training data, leading to more robust and reliable model outputs.

 ## Setup

```

git clone https://github.com/cyp-jlu-ai/BA-LoRA.git
cd BA-LoRA
conda create --name ba-lora python=3.9
pip install -r requirements.txt

```

## Usage

```

python /scripts/ba-lora.sh

```
