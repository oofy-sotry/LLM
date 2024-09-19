import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from huggingface_hub import login

# 모델 및 데이터셋 설정
model_name = "/home/oofy/Documents/LangChain/model/Llama-2-7b-hf"
dataset_name = "HAERAE-HUB/Korean-Human-Judgements"
new_model = "jeunghyen/llama-2-ko-7b-hf-1"

# LoRA 설정
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# 4-bit precision 설정
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# 훈련 설정
output_dir = "./results"
num_train_epochs = 1
fp16 = False
bf16 = False  # ROCm에서는 bf16 지원 여부에 따라 조정
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

# ROCm 지원 확인 및 device_map 설정
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7:
    device_map = "auto"
    fp16 = True  # ROCm에서 fp16 사용
else:
    device_map = {"": 0}  # 모델을 첫 번째 GPU에 로드

login(token="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")

# 데이터셋 로드

dataset = load_dataset(dataset_name, split="train")

# 모델 계산에 사용될 데이터 타입 결정
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# 기본 모델 로드
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map,
    torch_dtype=torch.float16 if fp16 else torch.float32  # ROCm에서 fp16 사용
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
    fp16=fp16,
    bf16=bf16,
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
    return_dict=True,
    torch_dtype=torch.float16 if fp16 else torch.float32
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
