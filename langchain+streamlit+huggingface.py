from transformers import pipeline
import streamlit as st

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-hf")

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
    bot_response = pipe(user_input, max_length=100, num_return_sequences=1)[0]['generated_text']
    add_to_chat_history(user_input, bot_response)
    st.session_state.user_input = ""

display_chat_history()
