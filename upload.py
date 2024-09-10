# Hugging Face upload 하기 위해서 필요한 코드

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from huggingface_hub import login

# Hugging Face Hub 로그인
login(token="발급받은 토큰 사용")

# 모델 저장 경로와 이름 설정
model_save_dir = "모델이 저장되어져 있는 폴더 위치"
MODEL_SAVE_REPO = "branch 이름"

# 모델 및 토크나이저 로드
model = AutoModelForCausalLM.from_pretrained(model_save_dir)
tokenizer = AutoTokenizer.from_pretrained(model_save_dir)

# 모델 및 토크나이저 저장
model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)

# Hugging Face Hub에 모델 및 토크나이저 업로드
model.push_to_hub(
    repo_id=MODEL_SAVE_REPO,
    use_temp_dir=True,
    use_auth_token="토큰"
)
tokenizer.push_to_hub(
    repo_id=MODEL_SAVE_REPO,
    use_temp_dir=True,
    use_auth_token="토큰"
)
