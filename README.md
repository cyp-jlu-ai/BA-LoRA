#BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models

## Introduction
Large language models (LLMs) have demonstrated remarkable proficiency across various natural language processing (NLP) tasks. However, adapting LLMs to downstream applications requires computationally intensive and memory-demanding fine-tuning procedures. To alleviate these burdens, parameter-efficient fine-tuning (PEFT) techniques have emerged as a promising approach to tailor LLMs with minimal computational overhead. While PEFT methods offer substantial advantages, they do not fully address the pervasive issue of bias propagation from pre-training data. This work introduces Bias-Alleviating Low-Rank Adaptation (BA-LoRA), a novel PEFT method designed to counteract bias inheritance. BA-LoRA incorporates three distinct regularization terms: (1) a consistency regularizer, (2) a diversity regularizer, and (3) a singular value decomposition regularizer. These regularizers aim to enhance the models' consistency, diversity, and generalization capabilities during fine-tuning. We conduct extensive experiments on natural language understanding (NLU) and natural language generation (NLG) tasks using prominent LLMs such as LLaMA, Mistral, and Gemma. The results demonstrate that BA-LoRA outperforms LoRA and its state-of-the-art variants. Moreover, our method effectively mitigates the adverse effects of pre-training bias, leading to more reliable and robust model outputs.

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/cyp-jlu-ai/BA-LoRA.git
    ```

2. Navigate to the directory:
    ```bash
    cd BA-LoRA
    ```

3. Create and activate a conda environment:
    ```bash
    conda create --name ba-lora python=3.9
    conda activate ba-lora
    ```

4. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the script:
```bash
sh scripts/ba-lora.sh

```

## Citation

If you find this project useful in your research or work, please consider citing it:

```
@article{chang2024bias,
  title={Bias-Aware Low-Rank adaptation: Mitigating catastrophic inheritance of large language models},
  author={Chang, Yupeng and Chang, Yi and Wu, Yuan},
  journal={arXiv preprint arXiv:2408.04556},
  year={2024}
}
```
