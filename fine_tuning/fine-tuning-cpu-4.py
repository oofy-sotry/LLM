import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login

# Hugging Face Hub 로그인
login(token="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")

# 모델 및 데이터셋 설정
model_name = "jeunghyen/llama-2-ko-7b-3"
dataset_name = "psyche/korean_idioms"
dataset_config_name = "default"
new_model = "jeunghyen/llama-2-ko-7b-4"

# LoRA 설정
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

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
optim = "adamw_hf"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = False
save_steps = 1000
logging_steps = 25

# 시퀀스의 최대 길이 및 패킹 여부
max_seq_length = 512
packing = False

# device_map 설정 (CPU로 설정)
device_map = {"": "cpu"}

# 데이터셋 로드
try:
    dataset = load_dataset(dataset_name, dataset_config_name, split="train")
    print(f"Dataset '{dataset_name}' loaded successfully.")
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    raise

# 데이터셋 필드 확인
datafield = dataset.column_names
print(f"Dataset fields: {datafield}")

# 텍스트 필드가 있는지 확인 후 할당
if 'text' in datafield:
    dataset_text_field = 'text'
else:
    # 텍스트 필드가 없으면 첫 번째 필드를 기본값으로 설정
    dataset_text_field = datafield[0]
    print(f"Using '{dataset_text_field}' as the dataset text field.")

# 모델 로드 (CPU에서 로드)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)
model.config.use_cache = False

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 데이터셋 토큰화 함수 정의
def tokenize_function(examples):
    # 텍스트 필드에서 데이터 추출
    texts = examples[dataset_text_field]

    # 텍스트가 리스트 형식인지 확인
    if not isinstance(texts, list):
        raise ValueError("The input to the tokenizer must be a list of strings.")
    
    # 각 텍스트가 문자열인지 확인하고, 숫자형 데이터는 문자열로 변환
    processed_texts = []
    for text in texts:
        if isinstance(text, str):
            processed_texts.append(text)
        elif isinstance(text, int):
            # 숫자형 데이터를 문자열로 변환
            processed_texts.append(str(text))
        else:
            raise ValueError(f"Each element in the batch must be a string or an integer, but got {type(text)} instead.")
    
    # 텍스트를 토큰화
    return tokenizer(
        processed_texts,  # 텍스트 리스트
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
    )

# 데이터셋 토큰화
try:
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    print("Dataset tokenized successfully.")
except Exception as e:
    print(f"An error occurred during tokenization: {e}")
    raise

# 'label' 필드를 'labels'로 변경 (필요할 경우)
if 'label' in tokenized_dataset.column_names:
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")

# LoRA 설정 로드
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

# SFTConfig로 설정값 전달
sft_config = SFTConfig(
    dataset_text_field=dataset_text_field,
    max_seq_length=max_seq_length,
    packing=packing,
    output_dir=output_dir
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
    report_to="none"
)

# SFTTrainer 설정
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_dataset,  # 토큰화된 데이터셋 사용
    tokenizer=tokenizer,
    args=training_arguments,
    peft_config=peft_config,
    dataset_text_field=dataset_text_field,
    max_seq_length=max_seq_length,
    packing=packing
)

# 모델 훈련
trainer.train()

# 훈련된 모델 저장 경로
model_save_dir = "./trained_model_4"

# 훈련된 모델 저장
trainer.model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)

# LoRA 가중치 병합 및 모델 저장
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True
)
peft_model = get_peft_model(base_model, peft_config)

# 병합 전 활성 어댑터 확인 및 수동 설정
peft_model.set_active_adapters(["default"])  # 활성 어댑터를 명시적으로 설정

# 병합된 모델로부터 최종 모델 저장
merged_model = peft_model.merge_and_unload()

# Hugging Face Hub에 모델 및 토크나이저 푸시
merged_model.push_to_hub(new_model)
tokenizer.push_to_hub(new_model)
