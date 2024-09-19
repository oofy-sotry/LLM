from langchain_community.llms import Ollama

# Ollama LLM 객체 생성
llm = Ollama(model="llama2")

# invoke 메서드를 사용하여 LLM 호출
response = llm.invoke("Hi")

# 결과 출력
print(response)