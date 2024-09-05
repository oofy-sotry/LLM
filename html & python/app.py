from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

llm = Ollama(model="llama2")
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('user_prompt')

    try:
        response = chain.invoke({'input': user_input})
        print(response)
        return jsonify({'response': response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'API 호출 중 오류 발생'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
