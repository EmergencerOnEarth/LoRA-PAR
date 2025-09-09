import subprocess
import argparse
        
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, concatenate_datasets, Dataset
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer

import os
import numpy as np
import pandas as pd
import re
import random
import zipfile
import logging
from typing import Dict, Sequence, Union, List
from tqdm import tqdm
import math
import wandb


alpha = 1 #NOT lora_alpha. the ratio of overlapping area of parameters used during SFT, a float number between 0 and 1
beta = 1 #the ratio of overlapping area of parameters used during RL, a float number between 0 and 1
wandb_project = 'pipeline'
basemodel = './autodl-tmp/Llama-2-7b-chat-hf'
lora_module = 'qkvgud'
lora_rank = 8
lora_alpha = 64
dataset_name = 'gsm8k'
importance_dir = './importance'
importance_threshold = 0.9
sft_output_dir = './pipeline_weight/sft'
rl_output_dir = './pipeline_weight/rl'
rl_run_name = 'pipeline'


def get_lora_config(target_modules=["q_proj","k_proj","v_proj"],
    r=8, lora_alpha=32, lora_dropout=0.1):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM"
    )
    return lora_config

def load_llama2_with_lora(
    lora_config,
    base_model_name,
    use_grad_ckpt=False
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16, 
        device_map='auto'
    ) 
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # ensure pad token is set
        model.config.pad_token_id = tokenizer.eos_token_id
    
    if use_grad_ckpt:
        model.config.use_cache = False 
        model.gradient_checkpointing_enable()
    else:
        model.config.use_cache = True    

    model = get_peft_model(model, lora_config)
    
    for name, param in model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
            
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            torch.nn.init.normal_(param.data, mean=0.0, std=0.01)

    return model, tokenizer

def mask_prompt_in_labels(
    input_ids: torch.Tensor,
    prompt_length: int
):
    labels = input_ids.clone()
    labels[:, :prompt_length] = -100
    return labels

def compute_lora_importance_2ndorder(
    model,
    dataloader,
    device="cuda",
    max_steps=None
):
    model.train()
    model.to(device)
    model.zero_grad()

    lora_params = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            shape_ = p.data.shape
            lora_params[name] = {
                "param": p, 
                "sum_grad_param": torch.zeros(shape_, device=device),
                "sum_sqr": torch.zeros(shape_, device=device)
            }
    
    step_count = 0
    for step, batch in enumerate(dataloader):
        for k in batch:
            batch[k] = batch[k].to(device)
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for name, buf in lora_params.items():
            p = buf["param"]
            if p.grad is not None:
                grad_p = p.grad
                # grad * p
                tmp = grad_p * p.data  
                buf["sum_grad_param"] += tmp
                # (grad*p)^2
                buf["sum_sqr"] += tmp.pow(2)

        model.zero_grad()
        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            break
    
    importance = {}
    for name, buf in lora_params.items():
        s1 = buf["sum_grad_param"]
        s2 = buf["sum_sqr"]
        # formula: | s1 - 0.5 * s2 |
        # imp = torch.abs(s1 - 0.5 * s2)
        imp = torch.abs(s1 - 0.5 * s2) / step_count
        importance[name] = imp.detach().cpu()
    
    return importance

def parse_lora_name(param_name: str):
    pattern_layer = re.compile(r"layers\.(\d+)\.")
    match_layer = pattern_layer.search(param_name)
    L = int(match_layer.group(1)) if match_layer else -1
    
    # 投影类型
    if ".q_proj." in param_name:
        M = "Q"
    elif ".k_proj." in param_name:
        M = "K"
    elif ".v_proj." in param_name:
        M = "V"
    elif ".up_proj." in param_name:
        M = "U"
    elif ".down_proj." in param_name:
        M = "D"
    elif ".gate_proj." in param_name:
        M = "G"
    else:
        M = "O"  # or "Other"
    
    # A 或 B
    if "lora_A" in param_name:
        m_in = "A"
    elif "lora_B" in param_name:
        m_in = "B"
    else:
        m_in = "?"
    
    H = -1
    
    return L, H, M, m_in

def importance_to_dataframe(
    importance_dict: dict,
    system_label="system1"
):
    rows = []
    global_index = 0
    
    for name, imp_tensor in importance_dict.items():
        L, H, M, m_in = parse_lora_name(name)
        # flatten
        imp_flat = imp_tensor.view(-1)
        for val in imp_flat:
            rows.append({
                "index": global_index,
                "L": L,
                "H": H,
                "M": M,
                "m": m_in,
                "E": val.item(),
                "system": system_label 
            })
            global_index += 1
    
    df = pd.DataFrame(rows, columns=["index","L","H","M","m","E","system"])
    return df

def zip_and_delete(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    zip_path = file_path + '.zip'

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(file_path, arcname=os.path.basename(file_path))

    os.remove(file_path)
    print(f"文件已压缩为: {zip_path}，原文件已删除。")
    
def collate_fn(examples, max_length=4096):
    input_ids_list = []
    labels_list = []
    for prompt, resp in examples:
        full_text = prompt + resp
        encoded = tokenizer(
            full_text, 
            add_special_tokens=False,
            truncation=True,
            max_length=max_length 
        )
        prompt_enc = tokenizer(
            prompt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length//2
        )
        resp_enc = tokenizer(
            resp,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - len(prompt_enc["input_ids"])
        )
        
        input_ids = prompt_enc["input_ids"] + resp_enc["input_ids"]
        labels = [-100]*len(prompt_enc["input_ids"]) + resp_enc["input_ids"]

        input_ids_list.append(torch.tensor(input_ids))
        labels_list.append(torch.tensor(labels))

    max_len_in_batch = max(len(x) for x in input_ids_list)
    batch_input = torch.full((len(examples), max_len_in_batch), tokenizer.pad_token_id, dtype=torch.long)
    batch_labels = torch.full((len(examples), max_len_in_batch), -100, dtype=torch.long)
    for i, (inp, lbl) in enumerate(zip(input_ids_list, labels_list)):
        seq_len = len(inp)
        batch_input[i, :seq_len] = inp
        batch_labels[i, :seq_len] = lbl

    return {
        "input_ids": batch_input,
        "labels": batch_labels
    }

def load_text_list(df, prompt_template):
    system_texts = list()

    for t in df.itertuples():
        prompt = prompt_template.format(t.question)        
        system_texts.append((prompt, t.answer))
    return system_texts

def find_cutpoint(df, column, threshold=0.85):
    df_sorted = df.sort_values(by=column, ascending=False)
    values = df_sorted[column].values
    cumsum = np.cumsum(values)
    total = values.sum()
    cutoff_index = np.searchsorted(cumsum, threshold * total)
    cutpoint_value = values[cutoff_index]
    return cutpoint_value


def freeze_elements_with_mask(param: torch.nn.Parameter, bool_mask: torch.Tensor):
    assert param.shape == bool_mask.shape, "mask 形状必须与 param 一致"
    
    param.requires_grad_(True)
    param._freeze_mask = bool_mask

    def grad_mask_hook(grad):
        return grad.masked_fill(~bool_mask, 0.0)

    handle = param.register_hook(grad_mask_hook)
    param._freeze_mask_hook_handle = handle

def apply_freeze_mask_from_dataframe(model, df, col_name="sys1_activated", device=None):
    import numpy as np

    activated_array = df[col_name].values
    total_len = activated_array.shape[0]

    hook_handles = []
    current_offset = 0  
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        numel = param.numel()
        sub_arr = activated_array[current_offset : current_offset + numel]
        
        if sub_arr.shape[0] != numel:
            raise ValueError(f"DataFrame中可用数据长度不足，param={name}, "
                             f"需要 {numel} 个元素，但剩余只有 {sub_arr.shape[0]}。")
        
        bool_mask_np = (sub_arr == 1).reshape(param.shape)
        
        mask_device = device if device is not None else param.device
        bool_mask_torch = torch.as_tensor(bool_mask_np, dtype=torch.bool, device=mask_device)
        
        freeze_elements_with_mask(param, bool_mask_torch)

        hook_handles.append(param._freeze_mask_hook_handle)
        
        current_offset += numel
    
    if current_offset != total_len:
        print(f"warning：DataFrame has {total_len} 行，but only {current_offset} lines are in used.")
    
    print(f"[apply_freeze_mask_from_dataframe] Done. Used {current_offset} elements from '{col_name}'.")
    return hook_handles

def remove_all_freeze_hooks(model):
    for name, param in model.named_parameters():
        handle = getattr(param, "_freeze_mask_hook_handle", None)
        if handle is not None:
            handle.remove()
            del param._freeze_mask_hook_handle
        
        if hasattr(param, "_freeze_mask"):
            del param._freeze_mask
    
    print("[remove_all_freeze_hooks] All element-wise freeze hooks have been removed.")


def stats(model):
    trainable_count = 0
    total_count = 0
    for name, p in model.named_parameters():
        total_count += p.numel()
        if p.requires_grad:
            if hasattr(p, "_freeze_mask"):
                trainable_count += p._freeze_mask.sum().item()
            else:
                trainable_count += p.numel()
    return trainable_count, total_count



def build_instruction_dataset(data_path: Union[List[str],str],
                tokenizer: transformers.PreTrainedTokenizer,
                max_seq_length: int, data_cache_dir = None,
                preprocessing_num_workers = None,
                ):

    def tokenization(examples):
        sources = []
        targets = []
        prompt = sft_prompt_template #PROMPT_TEMPLATE
        
        for instruction, input, output in zip(examples['instruction'], examples['input'], examples['output']):
            if input is not None and input !="":
                instruction = instruction+'\n'+input
            # source = prompt.format_map({'instruction':instruction})
            source = prompt.format(instruction)
            
            # #XML
            # arr = output.split('####')
            # reasoning_process = arr[0].strip()
            # answer = arr[1].strip()
            # response = '{}\n<answer>\n{}\n</answer>'.format(reasoning_process, answer)
            # oringal
            response = output
            target = f"{response}{tokenizer.eos_token}"
            
            sources.append(source)
            targets.append(target)

        tokenized_sources = tokenizer(sources,return_attention_mask=False)
        tokenized_targets = tokenizer(targets,return_attention_mask=False,add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        all_attention_masks = []
        for s,t in zip(tokenized_sources['input_ids'],tokenized_targets['input_ids']):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            attention_masks = torch.LongTensor([1] * len(s + t))[:max_seq_length]
            assert len(input_ids) == len(labels)
            assert len(input_ids) == len(attention_masks)
            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_masks)
            
        results = {'input_ids':all_input_ids, 'labels': all_labels, 'attention_mask': all_attention_masks}
        return results


    logging.warning("building dataset...")
    all_datasets = []
    if not isinstance(data_path,(list,tuple)):
        data_path = [data_path]
    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(data_cache_dir,os.path.basename(file).split('.')[0]+f"_{max_seq_length}")
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f'training datasets-{file} has been loaded from disk')
        except Exception:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=cache_path)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction","input","output"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format('torch')
        all_datasets.append(processed_dataset['train'])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        # input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            # attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
            attention_mask=attention_mask
        )

def set_lora_dropout(model, new_dropout: float):
    for name, module in model.named_modules():
        if hasattr(module, "lora_dropout") and isinstance(module.lora_dropout, torch.nn.Dropout):
            module.lora_dropout.p = new_dropout

    if hasattr(model, "peft_config"):
        if "default" in model.peft_config:
            model.peft_config["default"].lora_dropout = new_dropout

    print(f"[set_lora_dropout] LoRA dropout 已被修改为 {new_dropout}.")
    

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore

def format_question(question, test_sft_mdoel, apply_template=True, test_base_model=False):
    if test_base_model:
        # base_prompt_template.format
        question = base_prompt_template.format(question)
    else:
        if test_sft_mdoel:
            question = sft_prompt_template.format(question)
        else:
            if apply_template:
                t = [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': question}
                ]
                question = tokenizer.apply_chat_template(t, tokenize=False)
            else:
                question = rl_prompt_template.format(question)
    return question

def generate_answer(question, test_sft_mdoel):
    prompt_text = format_question(question, test_sft_mdoel)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    generation_args = dict(
        max_new_tokens=512,
        temperature=0.0,
        do_sample=False,
        top_p=0.95,
        # etc.
    )
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **generation_args
        )
    # 解码
    answer_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer_text

def extract_final_number(output_str):
    pattern = r"####\s*(-?\d+)"
    matches = re.findall(pattern, output_str)
    if matches:
        return int(matches[-1]) 
    else:
        return None

def extract_final_answer_plus(output_str):
    pattern = r"<answer>(.*?)</answer>"
    # pattern = r"<answer>\s*(-?\d+)</answer>"
    matches = re.findall(pattern, output_str, re.DOTALL)  
    if matches:
        try:
            return int(matches[-1])  
        except:
            return None
    else:
        return None
    
def batch_generate_and_eval_plus(
    questions,
    ref_answers,
    test_sft_mdoel, 
    batch_size=4,
    max_new_tokens=256
):
    assert len(questions) == len(ref_answers), "问题与参考答案长度不一致"
    total_samples = len(questions)

    correct_so_far = 0
    processed_so_far = 0

    total_batches = math.ceil(total_samples / batch_size)

    pbar = tqdm(total=total_batches, desc="Generating + Evaluating")

    for start_idx in range(0, total_samples, batch_size):
        batch_questions = questions[start_idx : start_idx + batch_size]
        batch_ref_answers = ref_answers[start_idx : start_idx + batch_size]

        batch_prompt_texts = [format_question(q, test_sft_mdoel) for q in batch_questions]
        inputs = tokenizer(
            batch_prompt_texts,
            return_tensors="pt",
            padding=True,          
            truncation=True
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False
            )

        batch_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for pred_str, ref_str in zip(batch_preds, batch_ref_answers):
            if test_sft_mdoel:
                gt_num = extract_final_number(ref_str)
                
                # pred_num = extract_final_answer_plus(pred_str)
                pred_num = extract_final_number(pred_str)
                # print('{}---{}'.format(gt_num, pred_num))
                if gt_num is not None and pred_num == gt_num:
                    # print('========')
                    correct_so_far += 1
            else:
                gt_num = ref_str
                pred_num = extract_final_answer_plus(pred_str)
                if gt_num is not None and ((pred_num == gt_num) or (str(pred_num) == str(gt_num))):
                    correct_so_far += 1

        processed_so_far += len(batch_questions)

        partial_acc = correct_so_far / processed_so_far

        pbar.set_postfix({'acc': f"{partial_acc:.4f}"})
        pbar.update(1)

    pbar.close()
    final_acc = correct_so_far / processed_so_far
    return final_acc


# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses] 
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    try:
        matches = [re.match(pattern, r.strip(), flags=re.DOTALL) for r in responses] 
    except:
        matches = [re.match(pattern, r, flags=re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


def activate_in_pct(df, threshold=0.85, alpha=1.0, beta=1.0):
    df_new = df.copy()
    
    df_new["sys1_activated"] = 0
    df_new["sys2_activated"] = 0

    #  1a) sys1
    s1_sum = df_new["sys1_E_normalized"].sum()
    df_s1_sorted = df_new.sort_values("sys1_E_normalized", ascending=False)
    cumsum_s1 = 0.0
    s1_list = []
    for idx, row in df_s1_sorted.iterrows():
        cumsum_s1 += row["sys1_E_normalized"]
        s1_list.append(idx)
        if cumsum_s1 >= threshold * s1_sum:
            break
    s1Set = set(s1_list)
    
    #  1b) sys2
    s2_sum = df_new["sys2_E_normalized"].sum()
    df_s2_sorted = df_new.sort_values("sys2_E_normalized", ascending=False)
    cumsum_s2 = 0.0
    s2_list = []
    for idx, row in df_s2_sorted.iterrows():
        cumsum_s2 += row["sys2_E_normalized"]
        s2_list.append(idx)
        if cumsum_s2 >= threshold * s2_sum:
            break
    s2Set = set(s2_list)

    # ============ 2) s1only, s2only, shared ============ 
    s1only = s1Set - s2Set
    s2only = s2Set - s1Set
    shared = s1Set.intersection(s2Set)

    # ============ 3) s1only => sys1=1, sys2=0 ============
    df_new.loc[list(s1only), "sys1_activated"] = 1
    
    # ============ 4) s2only => sys1=0, sys2=1 ============ 
    df_new.loc[list(s2only), "sys2_activated"] = 1

    # ============ 5) alpha / beta strategy in shared part ============ 
    shared_df = df_new.loc[list(shared)].copy()

    shared_df_s1 = shared_df.sort_values("sys1_E_normalized", ascending=False)
    n_sh = len(shared_df_s1)
    topN_s1 = int(alpha * n_sh)
    s1_idx = shared_df_s1.index[:topN_s1]
    df_new.loc[s1_idx, "sys1_activated"] = 1

    shared_df_s2 = shared_df.sort_values("sys2_E_normalized", ascending=False)
    topN_s2 = int(beta * n_sh)
    s2_idx = shared_df_s2.index[:topN_s2]
    df_new.loc[s2_idx, "sys2_activated"] = 1

    return df_new


#wandb setting
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]=wandb_project

# save your trained model checkpoint to wandb
# os.environ["WANDB_LOG_MODEL"]="checkpoint"

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"


#random seed
SEED = 41
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

#basemodel
model_name = basemodel

if lora_module == 'qkvgud':
    target_modules=["q_proj","k_proj","v_proj","gate_proj","up_proj","down_proj"]
elif lora_module == 'qkv':
    target_modules=["q_proj","k_proj","v_proj"]
elif lora_module == 'gud':
    target_modules=["gate_proj","up_proj","down_proj"]

# lora setting
r = lora_rank
lora_dropout_for_calculate_importance = 0 
# lora_alpha = 64 

## dataset
if dataset_name == 'gsm8k':
    data_file = './dataset_samples/df_gsm8k_all_classified_voting_3-5.csv'

# importance_dir = './importance_llama2/'
# importance_threshold = 0.9 
IGNORE_INDEX = -100
dropout_for_training = 0.05

# sft_output_dir = "./llama2-7b-gsm8k-sft_qkvgud_freezed/"


sft_max_seq_length=512
sft_learning_rate=2e-4
sft_lr_scheduler_type="cosine"
sft_num_train_epochs=1
sft_per_device_train_batch_size=1
sft_gradient_accumulation_steps=32
sft_warmup_ratio=0.03
sft_logging_steps=1
sft_logging_first_step=True
sft_save_total_limit=3
report_to="wandb"
sft_evaluation_strategy="steps"
sft_eval_steps=50


sft_prompt_template = "{}</s>"


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

rl_prompt_template = '''
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

{}\n
'''


device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load model
print('loading base model and lora module: basemodel: {}, lora module: {}'.format(model_name, lora_module))
lora_config = get_lora_config(target_modules, r, lora_alpha, lora_dropout_for_calculate_importance)


model, tokenizer = load_llama2_with_lora(
    lora_config, 
    base_model_name=model_name,
    use_grad_ckpt=False
)
model.to(device)


# 2. importance calculation 
print('calculating importance of weights for 2 stages training process ...')
df_sampled = pd.read_csv(data_file)

df_sampled_1 = df_sampled[df_sampled['_result'] == 'System1']
df_sampled_2 = df_sampled[df_sampled['_result'] == 'System2']

df_sampled_1 = df_sampled_1.sample(100)
df_sampled_2 = df_sampled_2.sample(100)

system1_texts = load_text_list(df_sampled_1, sft_prompt_template)
system2_texts = load_text_list(df_sampled_2, sft_prompt_template)


#system1
print('calculating for system 1 ...')
system1_loader = DataLoader(system1_texts, batch_size=1, collate_fn=collate_fn)

sys1_importance = compute_lora_importance_2ndorder(model, system1_loader, device=device)
sys1_df = importance_to_dataframe(sys1_importance, system_label="system1")

#system2
print('calculating for system 2 ...')
system2_loader = DataLoader(system2_texts, batch_size=1, collate_fn=collate_fn)

sys2_importance = compute_lora_importance_2ndorder(model, system2_loader, device=device)
sys2_df = importance_to_dataframe(sys2_importance, system_label="system2")

column_name = "E" 
# L2 normalization
norm = np.linalg.norm(sys1_df[column_name], ord=2)

if norm == 0:
    sys1_df[column_name + "_normalized"] = sys1_df[column_name]
else:
    sys1_df[column_name + "_normalized"] = sys1_df[column_name] / norm

norm = np.linalg.norm(sys2_df[column_name], ord=2)

if norm == 0:
    sys2_df[column_name + "_normalized"] = sys2_df[column_name]
else:
    sys2_df[column_name + "_normalized"] = sys2_df[column_name] / norm


df_sys_importance = pd.DataFrame()
df_sys_importance['index'] = list(sys1_df['index'])
df_sys_importance['L'] = list(sys1_df['L'])
df_sys_importance['M'] = list(sys1_df['M'])
df_sys_importance['m'] = list(sys1_df['m'])
df_sys_importance['sys1_E'] = list(sys1_df['E'])
df_sys_importance['sys1_E_normalized'] = list(sys1_df['E_normalized'])
df_sys_importance['sys2_E'] = list(sys2_df['E'])
df_sys_importance['sys2_E_normalized'] = list(sys2_df['E_normalized'])


os.makedirs(importance_dir, exist_ok=True)
_path = os.path.join(importance_dir, 'importance_plus_{}'.format(os.path.basename(data_file)))
df_sys_importance.to_csv(_path)

zip_and_delete(_path)


#3. 
print('setting training ratio of overlapping parameters for 2 stages tuning process, alpha: {}, beta: {}'.format(alpha, beta))
df_sys_importance = activate_in_pct(df_sys_importance, threshold=importance_threshold, alpha=alpha, beta=beta)

df_activated = df_sys_importance[['index', 'sys1_activated', 'sys2_activated']]

df_activated.to_csv(
    os.path.join(importance_dir, 
                 'activation_{}_{}_r{}.csv'.format(dataset_name, lora_module, r)))


phase1_handles = apply_freeze_mask_from_dataframe(model, df_activated, col_name="sys1_activated")

trainable_count, total_count = stats(model)
print(f"Trainable: {trainable_count} / {total_count} = {100*trainable_count/total_count:.2f}%")


#4. SFT
data_dir = './dataset_samples/data_plus/{}/'.format(dataset_name)

train_dataset = build_instruction_dataset(
    data_path=os.path.join(data_dir, 'train/train.json'),
    tokenizer=tokenizer,
    max_seq_length=512,
    data_cache_dir=None,
    preprocessing_num_workers=1)

test_dataset = build_instruction_dataset(
    data_path=os.path.join(data_dir, 'test/test.json'),
    tokenizer=tokenizer,
    max_seq_length=512,
    data_cache_dir=None,
    preprocessing_num_workers=1)

data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

model.peft_config['default'].lora_dropout = dropout_for_training

# SFTConfig
sft_config = SFTConfig(
    max_seq_length=sft_max_seq_length,
    learning_rate=sft_learning_rate,
    lr_scheduler_type=sft_lr_scheduler_type,
    num_train_epochs=sft_num_train_epochs,
    per_device_train_batch_size=sft_per_device_train_batch_size,
    gradient_accumulation_steps=sft_gradient_accumulation_steps,
    warmup_ratio=sft_warmup_ratio,
    # bf16=True,
    fp16=True,
    logging_steps=sft_logging_steps,
    logging_first_step=sft_logging_first_step,
    save_total_limit=sft_save_total_limit,
    output_dir=sft_output_dir,
    report_to="wandb",
    seed=SEED,
    evaluation_strategy=sft_evaluation_strategy,
    eval_steps=sft_eval_steps
)

# 初始化 Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    args=sft_config,
    # peft_config=lora_config_for_training,
    data_collator=data_collator
)

# Train the model
trainer.train()

 
print("\n=== Remove freeze hooks after Phase 1 ===")
remove_all_freeze_hooks(model)

trainable_count, total_count = stats(model)
print(f"Trainable: {trainable_count} / {total_count} = {100*trainable_count/total_count:.2f}%")


#5. test SFT performace
dataset = load_dataset("gsm8k", "main")

test_data = dataset["test"]
print("Test samples:", len(test_data))

model.eval()

questions = [ex["question"] for ex in test_data]
ref_answers = [ex["answer"] for ex in test_data]

acc = batch_generate_and_eval_plus(questions, ref_answers, test_sft_mdoel=True, batch_size=16, max_new_tokens=256)
print(f"Final SFT test accuracy = {acc*100:.2f}%")


#6. RL
model.train()

apply_freeze_mask_from_dataframe(model, df_activated, col_name="sys2_activated")
trainable_count, total_count = stats(model)
# print(f"Trainable: {trainable_count} / {total_count} = {100*trainable_count/total_count:.2f}%")
print(f"Trainable: {trainable_count} / {total_count} = {100*trainable_count/total_count}%")

dataset = get_gsm8k_questions()

rl_learning_rate=5e-6
rl_adam_beta1 = 0.9
rl_adam_beta2 = 0.99
rl_weight_decay = 0.1
rl_warmup_ratio = 0.1
rl_lr_scheduler_type='cosine'
rl_logging_steps=1
rl_per_device_train_batch_size=8
rl_gradient_accumulation_steps=4
rl_num_generations=8 
rl_max_prompt_length=128 #256
rl_max_completion_length=256 #786
rl_num_train_epochs=1
rl_save_steps=100
rl_save_total_limit=3
rl_max_grad_norm=0.1

training_args = GRPOConfig(
    output_dir=rl_output_dir,
    run_name=rl_run_name,
    learning_rate=rl_learning_rate,
    adam_beta1=rl_adam_beta1,
    adam_beta2=rl_adam_beta2,
    weight_decay=rl_weight_decay,
    warmup_ratio=rl_warmup_ratio,
    lr_scheduler_type=rl_lr_scheduler_type,
    logging_steps=rl_logging_steps,
    bf16=True,
    per_device_train_batch_size=rl_per_device_train_batch_size,
    gradient_accumulation_steps=rl_gradient_accumulation_steps,
    num_generations=rl_num_generations,
    max_prompt_length=rl_max_prompt_length,
    max_completion_length=rl_max_completion_length,
    num_train_epochs=rl_num_train_epochs,
    # max_steps=1, #for test
    save_steps=rl_save_steps,
    save_total_limit=rl_save_total_limit,
    max_grad_norm=rl_max_grad_norm,
    report_to="wandb",
    log_on_each_node=False,
)


# use peft at your own risk; not working for me with multi-GPU training
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
    # peft_config=peft_config
)

try:
    trainer.train()
    
    print("\n=== Remove freeze hooks after Phase 2 ===")
    remove_all_freeze_hooks(model)
    
    model.save_pretrained(os.path.join(rl_output_dir, 'final_save'))
except:
    model.save_pretrained('./autodl-tmp/outputs/RL_weight/final_save/') #just a default dir


#7. test RL performace
model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

test_data = test_data.map(lambda x: { # type: ignore
    'prompt': [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': x['question']}
    ],
    'answer': extract_hash_answer(x['answer'])
})

questions = [ex["question"] for ex in test_data]
ref_answers = [ex["answer"] for ex in test_data]

acc = batch_generate_and_eval_plus(questions, ref_answers, test_sft_mdoel=False, batch_size=16, max_new_tokens=256)
print(f"Final SFT test accuracy = {acc*100:.2f}%")
      







