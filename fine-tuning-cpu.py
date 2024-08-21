import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import login

# 모델 및 데이터셋 설정
#model_name = "/home/oofy/Documents/LangChain/model/Llama-2-7b-hf"
model_name = "meta-llama/Llama-2-7b-hf"
dataset_name = "HAERAE-HUB/Korean-Human-Judgements"
new_model = "jeunghyen/llama-2-ko-7b-1"

# LoRA 설정
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# CPU 사용 설정 (4-bit 비활성화)
use_4bit = False

# 훈련 설정
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 1
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-6
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 1000  # 체크포인트 저장 스텝
logging_steps = 25

# 미리 정의되지 않은 변수 초기화
max_seq_length = 512  # 시퀀스의 최대 길이
packing = True  # 패킹 여부

# device_map 설정 (CPU로 설정)
device_map = {"": "cpu"}  # CPU에 로드

login(token="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")

# 데이터셋 로드
try:
    dataset = load_dataset(dataset_name, split="train")
    print(f"Dataset '{dataset_name}' loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")

# 기본 모델 로드 (CPU에서 로드)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device_map,  # CPU 사용
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32  # CPU에서 float32로 명시적 설정
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# LLaMA 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰을 EOS 토큰으로 설정
tokenizer.padding_side = "right"  # 패딩을 오른쪽에 추가

# LoRA 설정 로드
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# 훈련 파라미터 설정
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,  # CPU에서 fp16 사용 안 함
    bf16=bf16,  # CPU에서 bf16 사용 안 함
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

# 모델 훈련
trainer.train()

# 훈련된 모델 저장
trainer.model.save_pretrained(new_model)

# 기본 모델과 통합된 LoRA 가중치 저장
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()

# 사전 훈련된 토크나이저 로드 및 설정
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Hugging Face Hub에 푸시
login(token="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")  # Hugging Face API 토큰으로 로그인
model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
