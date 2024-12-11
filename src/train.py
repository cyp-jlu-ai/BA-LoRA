# trainer.py

import math
import copy
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List, Literal

import torch
import transformers
from typing import Dict, Union, Any
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, PeftModel
import torch.nn as nn
import numpy as np
from torch.nn.functional import kl_div, softmax
import torch.linalg as linalg
from datasets import load_metric

IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)

@dataclass
class BA_LoRA_TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "Dataset split to use."})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    lora_r: Optional[int] = field(default=None, metadata={"help": "The rank of the adapter. When passing `None` and `adapter_name_or_path` is also `None`, full fine-tuning is used."})
    init_lora_weights: Literal[True, "pissa"] = field(default=True, metadata={"help": "Passing True results in the LoRA initialization. Passing `pissa` results in PiSSA initialization."})
    task_type: Literal["nlu", "nlg"] = field(default="nlu", metadata={"help": "Type of task: 'nlu' or 'nlg'."})
    k: int = field(default=3, metadata={"help": "Number of top singular values for SVD regularization."})
    lambda1: float = field(default=1e-4, metadata={"help": "Weight for consistency regularization."})
    lambda2: float = field(default=3e-4, metadata={"help": "Weight for diversity regularization."})
    lambda3: float = field(default=1e-4, metadata={"help": "Weight for SVD regularization."})

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Tokenizes a list of strings using the provided tokenizer.

    Args:
        strings (Sequence[str]): List of input strings.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.

    Returns:
        Dict: Dictionary containing input_ids, labels, and their lengths.
    """
    tokenized_list = [tokenizer(text, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True) for text in strings]
    input_ids = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(input_ids=input_ids, labels=input_ids.copy(), input_ids_lens=input_ids_lens, labels_lens=input_ids_lens.copy())

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Preprocesses sources and targets by concatenating them and preparing labels.

    Args:
        sources (Sequence[str]): Source instructions.
        targets (Sequence[str]): Target responses.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.

    Returns:
        Dict: Dictionary containing input_ids and labels.
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

class CustomTrainer(Trainer):
    def __init__(self, *args, k=3, lambda1=1e-4, lambda2=3e-4, lambda3=1e-4, pre_trained_model=None, task_type="nlu", **kwargs):
        """
        Custom Trainer implementing BA-LoRA regularizations.

        Args:
            *args: Variable length argument list for the base Trainer.
            k (int): Number of top singular values for SVD regularization.
            lambda1 (float): Weight for consistency regularization.
            lambda2 (float): Weight for diversity regularization.
            lambda3 (float): Weight for SVD regularization.
            pre_trained_model (nn.Module, optional): Pre-trained model for consistency regularization.
            task_type (str): Type of task, either "nlu" or "nlg".
            **kwargs: Arbitrary keyword arguments for the base Trainer.
        """
        super().__init__(*args, **kwargs)
        self.k = k
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.pre_trained_model = pre_trained_model
        self.task_type = task_type

        if self.pre_trained_model:
            self.pre_trained_model.eval()
            for param in self.pre_trained_model.parameters():
                param.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Computes the total loss including task loss and BA-LoRA regularization losses.

        Args:
            model (nn.Module): The model being trained.
            inputs (Dict): Input batch.
            return_outputs (bool): Whether to return model outputs.

        Returns:
            torch.Tensor or (torch.Tensor, Dict): Total loss and optionally model outputs.
        """
        outputs = model(**inputs)
        task_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # Initialize regularization loss
        reg_loss = 0.0

        if self.pre_trained_model:
            with torch.no_grad():
                pre_trained_outputs = self.pre_trained_model(**inputs)

            if self.task_type == "nlu":
                # NLU: Use logits for consistency and diversity regularization
                pre_logits = pre_trained_outputs.logits
                fine_logits = outputs.logits

                # Consistency Regularization: MSE between normalized logits
                pre_norm = torch.nn.functional.normalize(pre_logits, p=2, dim=-1)
                fine_norm = torch.nn.functional.normalize(fine_logits, p=2, dim=-1)
                consistency_loss = torch.nn.functional.mse_loss(fine_norm, pre_norm)

                # Diversity Regularization: Covariance loss on logits
                batch_size, dim = fine_logits.size()
                fine_centered = fine_logits - fine_logits.mean(dim=0)
                cov = torch.matmul(fine_centered.T, fine_centered) / (batch_size - 1)
                # Extract off-diagonal elements
                off_diagonal = cov - torch.diag(torch.diag(cov))
                diversity_loss = torch.sum(off_diagonal ** 2) / dim

                # Singular Value Decomposition Regularization on logits
                svd_loss = self.svd_regularization(fine_logits)

            elif self.task_type == "nlg":
                # NLG: Use probability distributions for consistency and diversity regularization
                pre_probs = torch.nn.functional.softmax(pre_trained_outputs.logits, dim=-1)
                fine_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                # Consistency Regularization: KLD between probability distributions
                consistency_loss = torch.nn.functional.kl_div(fine_probs.log(), pre_probs, reduction='batchmean')

                # Diversity Regularization: Entropy maximization
                entropy = -torch.sum(fine_probs * torch.log(fine_probs + 1e-10), dim=-1).mean()
                diversity_loss = -entropy  # Maximize entropy

                # Singular Value Decomposition Regularization on logits
                svd_loss = self.svd_regularization(outputs.logits)

            else:
                raise ValueError("Unsupported task type. Choose 'nlu' or 'nlg'.")

            # Weighted sum of regularization losses
            reg_loss = (
                self.lambda1 * consistency_loss +
                self.lambda2 * diversity_loss +
                self.lambda3 * svd_loss
            )

        # Total loss
        total_loss = task_loss + reg_loss

        return (total_loss, outputs) if return_outputs else total_loss

    def svd_regularization(self, logits):
        """
        Computes Singular Value Decomposition (SVD) regularization loss.

        Args:
            logits (torch.Tensor): Logits from the fine-tuned model.

        Returns:
            torch.Tensor: SVD regularization loss.
        """
        # Ensure the logits are 2D
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), -1)

        # Compute SVD
        u, s, v = linalg.svd(logits, full_matrices=False)
        top_k_singular = s[:, :self.k]
        sum_top_k = top_k_singular.sum(dim=1)
        sum_all = s.sum(dim=1) + 1e-10  # Add epsilon to prevent division by zero
        ratio = sum_top_k / sum_all

        # Objective is to maximize ratio, hence minimize negative ratio
        svd_loss = -ratio.mean()

        return svd_loss

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collates a batch of instances into tensors.

        Args:
            instances (Sequence[Dict]): List of input instances.

        Returns:
            Dict[str, torch.Tensor]: Batch dictionary with input_ids, labels, and attention_mask.
        """
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer, query, response):
    """
    Tokenizes and preprocesses training data.

    Args:
        examples (Dict): Batch of examples from the dataset.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to use.
        query (str): Field name for the instruction/input.
        response (str): Field name for the response/output.

    Returns:
        Dict: Dictionary containing input_ids and labels.
    """
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train():
    """
    Main training function.
    Parses arguments, initializes models and trainer, and starts training.
    """
    parser = transformers.HfArgumentParser(BA_LoRA_TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Load the main model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map="auto",
    )

    # Apply LoRA adapters if specified
    if script_args.adapter_name_or_path is not None:
        print(f"Loading adapter weights from {script_args.adapter_name_or_path}")
        model = PeftModel.from_pretrained(
            model,
            script_args.adapter_name_or_path,
            is_trainable=True
        )
    elif script_args.lora_r is not None:
        print(f"Initializing LoRA with rank {script_args.lora_r}")
        lora_config = LoraConfig(
            r=script_args.lora_r,
            lora_alpha=script_args.lora_r,
            init_lora_weights=script_args.init_lora_weights,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    else:
        print("Performing full parameter fine-tuning.")

    # Freeze specific parameters
    for name, params in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            params.requires_grad = False
        elif params.requires_grad:
            print(f"Parameter to train: {name}")

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load and preprocess dataset
    raw_train_datasets = load_dataset(script_args.data_path, split=script_args.dataset_split)
    train_dataset = raw_train_datasets.map(
        lambda examples: train_tokenize_function(examples, tokenizer, script_args.dataset_field[0], script_args.dataset_field[1]),
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on train dataset"
    )

    # Initialize data collator
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    # Load pre-trained model for consistency regularization if applicable
    pre_trained_model = None
    if script_args.adapter_name_or_path is not None or script_args.lora_r is not None:
        # Assuming the pre-trained model is the same as the main model
        pre_trained_model = transformers.AutoModelForCausalLM.from_pretrained(script_args.model_name_or_path)
        pre_trained_model.eval()
        for param in pre_trained_model.parameters():
            param.requires_grad = False

    # Initialize the custom trainer
    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        **data_module,
        k=script_args.k,
        lambda1=script_args.lambda1,
        lambda2=script_args.lambda2,
        lambda3=script_args.lambda3,
        pre_trained_model=pre_trained_model,
        task_type=script_args.task_type
    )
    model.config.use_cache = False

    # Start training
    trainer.train()
    trainer.save_state()

    # Save the fine-tuned model
    model.save_pretrained(os.path.join(script_args.output_dir, 'ft'))

if __name__ == "__main__":
    train()
