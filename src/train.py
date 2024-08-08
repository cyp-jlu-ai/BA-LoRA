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
from datasets import load_metric

IGNORE_INDEX = -100
PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


class CustomTrainer(Trainer):
    # def compute_loss(self, model, inputs, return_outputs=False):

    #     outputs = model(**inputs)
    #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    #     return (loss, outputs) if return_outputs else loss
    def compute_loss(self, model, pretrain_model, inputs, lambda1, lambda2, lambda3, alpha, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        with torch.no_grad():
            pretrain_outputs = pretrain_model(**inputs)
        kl_loss = F.kl_div(F.log_softmax(outputs.logits, dim=-1), F.softmax(pretrain_outputs.logits, dim=-1), reduction='batchmean')
        
        entropy_loss = -torch.mean(torch.sum(F.softmax(outputs.logits, dim=-1) * F.log_softmax(outputs.logits, dim=-1), dim=-1))
        
        logits = outputs.logits.view(-1, outputs.logits.size(-1))
        _, s, _ = torch.svd(logits)
        top_k_singular_values = s[:k]
        svd_loss = -(torch.sum(top_k_singular_values) / torch.sum(s) + alpha * torch.var(top_k_singular_values))


        total_loss = loss + lambda1 * kl_loss + lambda2 * entropy_loss + lambda3 * svd_loss
        
        return (total_loss, outputs) if return_outputs else total_loss
    
    

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    adapter_name_or_path: Optional[str] = field(default=None)
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_split: str = field(default="train[:100000]", metadata={"help": "(`['train', 'test', 'eval']`):"})
    dataset_field: List[str] = field(default=None, metadata={"help": "Fields of dataset input and output."})
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."})
    lora_r: int = field(default=None, metadata={"help": "The rank of the adapter. When passing `None` and `adapter_name_or_path` is also `None`, full fine-tuning is used."})
    init_lora_weights: Literal[True, "pissa"] = field(default=True, metadata={"help": ("Passing True (default) results in the LoRA initialization. Passing `pissa` results in PiSSA initialization.")})

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    tokenized_list = [tokenizer(text, return_tensors="pt", padding="longest", max_length=tokenizer.model_max_length, truncation=True) for text in strings]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(input_ids=input_ids, labels=labels, input_ids_lens=input_ids_lens, labels_lens=labels_lens)

def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
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
    sources = [PROMPT.format_map(dict(instruction=instruction)) for instruction in examples[query]]
    targets = [f"{output}{tokenizer.eos_token}" for output in examples[response]]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

def train():
    parser = transformers.HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model = transformers.AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        device_map="auto",
    )

    if script_args.adapter_name_or_path is not None:
        print(f"Load {script_args.init_lora_weights} from {script_args.adapter_name_or_path}")
        model = PeftModel.from_pretrained(
            model,
            script_args.model_name_or_path,
            subfolder=script_args.adapter_name_or_path,
            is_trainable=True
        )
    elif script_args.lora_r is not None:
        print(f"Initialized {script_args.init_lora_weights} layers")
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
        print("Full Parameter Fine-Tuning")

    for name, params in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            params.requires_grad = False
        if params.requires_grad:
            print(name)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

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

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    trainer = CustomTrainer(
        model=model,
        tokenizer=tokenizer,
        args=script_args,
        **data_module
    )
    model.config.use_cache = False

    trainer.train()
    trainer.save_state()
    model.save_pretrained(os.path.join(script_args.output_dir, 'ft'))

if __name__ == "__main__":
    train()

