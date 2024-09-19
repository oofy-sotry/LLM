import streamlit as st
import random
from langchain_community.llms import Ollama

# 모델 초기화
llm = Ollama(model="llama2")

# 챗봇의 환영 메시지를 반환하는 함수
def get_response():
    responses = [
        "안녕하세요! 무엇을 도와드릴까요?",
        "오늘도 좋은 하루 되세요!",
        "저는 당신의 질문에 최선을 다해 답변드리겠습니다."
    ]
    return random.choice(responses)

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "has_greeted" not in st.session_state:
    st.session_state.has_greeted = False
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# 대화 기록에 사용자 입력과 봇 응답을 추가하는 함수
def add_to_chat_history(user_input, bot_response):
    if user_input:  # 사용자 입력이 있을 경우에만 기록 추가
        st.session_state.chat_history.append(f"**사용자**: {user_input}")
    st.session_state.chat_history.append(f"**봇**: {bot_response}")

# 대화 기록을 화면에 표시하는 함수
def display_chat_history():
    # 대화 기록을 업데이트할 플레이스홀더 생성
    for message in st.session_state.chat_history:
        st.write(message)

# 레이아웃 설정
st.title("llama2 사용 챗봇")

# 처음 시작 시 챗봇의 환영 메시지를 추가
if not st.session_state.has_greeted:
    # 챗봇의 환영 메시지 받기
    greeting_message = get_response()
    add_to_chat_history("", greeting_message)  # 빈 문자열로 사용자 입력을 표시하지 않음
    st.session_state.has_greeted = True  # 환영 메시지가 표시되었음을 기록

# 대화 기록 표시
display_chat_history()

# 사용자 입력 필드
user_input = st.text_input("질문을 입력하세요", value=st.session_state.user_input, key="user_input_placeholder", placeholder="질문을 입력한 후 Enter 키를 눌러보세요!")

# 사용자가 입력을 제출했을 때
if user_input and st.session_state.user_input != user_input:
    # LLM으로부터 응답받기
    bot_response = llm.invoke(user_input)

    # 대화 기록에 추가
    add_to_chat_history(user_input, bot_response)

    # 입력 필드 상태 업데이트
    st.session_state.user_input = ""  # 입력 후 필드 비우기

    # 대화 기록 다시 표시 (업데이트된 상태)
    display_chat_history()
