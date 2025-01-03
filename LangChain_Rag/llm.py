from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from langchain import PromptTemplate
import torch
import os
import requests
import json

# 0. HuggingFace Hub 로그인 (환경 변수로 토큰 가져오기)
# token = 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
login(token=os.getenv("HUGGINGFACE_TOKEN"))

print("----------------------------------------------------------------------------------------------------")
print("1번")
print("전달받은 api 확인")
api_url = "http://127.0.0.1:5000/process"
headers = {"Content-Type": "application/json"}

try:
    # API 요청
    response = requests.get(api_url, headers=headers)

    # API 응답 처리
    if response.status_code == 200:
        api_data = response.json()
        query = api_data.get("query", "")
        content_list = api_data.get("contents", [])
        content = "\n".join(content_list)
        print(f"Query: {query}")
        print(f"Content: {content}")
    else:
        print(f"API 요청 실패: {response.status_code}, {response.text}")
        query = ""
        content = ""
except Exception as e:
    print(f"API 호출 중 오류 발생: {e}")
    query = ""
    content = ""

print("----------------------------------------------------------------------------------------------------")
print("2번")
try:
    # LLM 모델 로드
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    prompt_template = PromptTemplate(
        input_variables=["query", "content"],
        template="질문: {query}\n답변: {content}"
    )

    prompt = prompt_template.format(query=query, content=content)

    inputs = tokenizer(prompt, return_tensors='pt', max_length=4096, truncation=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # LLM을 통해 최종 답변 생성
    try:
        generated_text = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.7,
            top_p=0.95,
            top_k=50, 
            no_repeat_ngram_size=2
        )

        decoded_output = tokenizer.decode(generated_text[0], skip_special_tokens=True)
        final_output = " ".join(decoded_output.split())
        print(f"최종 생성된 답변: {final_output}")
    except Exception as e:
        print(f"텍스트 생성 중 오류 발생: {e}")

except Exception as e:
    print(f"검색 중 오류 발생: {e}")
    context = "검색된 내용이 없습니다."

print("----------------------------------------------------------------------------------------------------")
