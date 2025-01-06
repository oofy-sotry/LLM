from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import PromptTemplate
import torch
# from huggingface_hub import login
# import os

def generate_answer_from_llm(query, pageContents):
    print("LLM 서버로 전달 완료")
    print("전달받은 query : " + query)
    print("전달받은 pageContents : " + str(pageContents))
    try:
    # LLM 모델 로드
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    except Exception as e:
        print(f"Error loading model: {e}")
        return "LLM 모델 로드 오류"

# 기존 설정
#    prompt_template = PromptTemplate(
#        input_variables=["query", "content"],
#        template="질문: {query}\n답변: {content}"
#    )

# prompt 설정 변경
#    prompt_template = PromptTemplate(
#        input_variables=["query", "content"],
#        template="LLM 답변(3문장 이내로 간단히 요약) : {content}"
#    )

# 모델에만 지시 전달하는 방식으로 변경
    prompt_template = PromptTemplate(
        input_variables=["query", "pageContents"],
        template="LLM 답변 : {pageContents}"
    )
    additional_instructions = "3문장 이내로 간단히 요약해 주세요. 영어는 빼주세요."
        

    pageContents = "\n".join(pageContents)

# 기존 promt 설정 방법    
#    prompt = prompt_template.format(query=query, content=content)
# 모델에만 지시하는 additional_instruction을 사용하기 위한 설정 변경
    prompt = prompt_template.format(query=query, pageContents=pageContents)
#    final_prompt = f"{prompt}"

    inputs = tokenizer(prompt, return_tensors='pt', max_length=4096, truncation=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # LLM을 통해 최종 답변 생성
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
    return decoded_output
