###################################################
## llama2 사용 
from langchain_community.llms import Ollama

llm = Ollama(model="llama2")

response = llm.invoke("지구의 자전 주기는?")

print(response)

# 답변
# The Earth's rotational period, also known as its sidereal day, is approximately 24 hours and 58 minutes. This is the time it takes for the Earth to rotate once on its axis, relative to the stars.
# However, the Earth's rotation is not perfectly uniform and can vary slightly due to a variety of factors, such as the Moon's gravitational pull and the effects of the tides. As a result, the length of a day on Earth can vary slightly over the course of a year, with longer days during the summer months in the Northern Hemisphere and shorter days during the winter months.
# In addition to its rotational period, the Earth also has a planetary rotation period, which is the time it takes for the Earth to rotate once on its axis relative to the Sun. This is approximately 365.24 days, which is the length of one year.

