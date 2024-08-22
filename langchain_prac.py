###################################################
## llama2 사용 
# from langchain_community.llms import Ollama

# llm = Ollama(model="llama2")

# response = llm.invoke("지구의 자전 주기는?")

# print(response)

# 답변
# The Earth's rotational period, also known as its sidereal day, is approximately 24 hours and 58 minutes. This is the time it takes for the Earth to rotate once on its axis, relative to the stars.
# However, the Earth's rotation is not perfectly uniform and can vary slightly due to a variety of factors, such as the Moon's gravitational pull and the effects of the tides. As a result, the length of a day on Earth can vary slightly over the course of a year, with longer days during the summer months in the Northern Hemisphere and shorter days during the winter months.
# In addition to its rotational period, the Earth also has a planetary rotation period, which is the time it takes for the Earth to rotate once on its axis relative to the Sun. This is approximately 365.24 days, which is the length of one year.




###################################################
## prompt + llama2 + output parser 사용
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
llm = Ollama(model="llama2")
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = chain.invoke({"input": "지구의 자전 주기는?"})

print(response)

# 답변
# Thank you for your question! The Earth's rotation period, also known as its rotational period, is approximately 24 hours. This is the time it takes for the Earth to
# rotate once on its axis. The Earth's rotation is slowing down over time due to the gravitational pull of the Moon and other celestial bodies, but it still takes
# about 24 hours for the planet to complete one rotation.