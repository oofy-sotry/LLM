import streamlit as st
import torch
import gc
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저 로드 - 허용 안나서 공개 모델 사용
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
용
# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_to_chat_history(user_input, bot_response):
    st.session_state.chat_history.append(f"사용자: {user_input}")
    st.session_state.chat_history.append(f"봇: {bot_response}")

def display_chat_history():
    for message in st.session_state.chat_history:
        st.write(message)

user_input = st.text_input("사용자 입력")

if user_input:
    # 사용자 입력을 토크나이즈하고 텐서로 변환
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # 모델로부터 응답 생성
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)  # max_length를 줄임
    
    # 응답을 디코딩하여 텍스트로 변환
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # 메모리 최적화: 사용된 GPU 메모리 해제
    torch.cuda.empty_cache()

    # CPU 메모리 최적화: 가비지 컬렉터 실행
    del input_ids, output  # 불필요한 텐서 삭제
    gc.collect()  # 가비지 컬렉터 실행

    # 대화 기록에 추가
    add_to_chat_history(user_input, bot_response)

    # 사용자 입력 초기화
    st.session_state.user_input = ""

# 대화 기록 표시
display_chat_history()
