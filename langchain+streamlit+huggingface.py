import streamlit as st
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저 로드
config = PeftConfig.from_pretrained("jeunghyen/llama-2-ko-7b-1")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "jeunghyen/llama-2-ko-7b-1")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

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
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    
    # 응답을 디코딩하여 텍스트로 변환
    bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    add_to_chat_history(user_input, bot_response)
    st.session_state.user_input = ""

display_chat_history()
