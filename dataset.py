from datasets import load_dataset

dataset_name = "HAERAE-HUB/Korean-Human-Judgements"
# 데이터셋을 Hugging Face Hub에서 로드
dataset = load_dataset(dataset_name, split="train")
