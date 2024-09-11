from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

app = Flask(__name__)
CORS(app)

# 모델과 토크나이저 로드
config = PeftConfig.from_pretrained("jeunghyen/llama-2-ko-7b-1")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "jeunghyen/llama-2-ko-7b-1")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")



llm = Ollama(model="llama2")
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

@app.route("/chat", methods=["POST"])
def chat():

    try:
        if request.get_json(silent=True) == None :
            prompt_text = "데이터가 들어오지 않아 질문이 없습니다."
        else :
            user_input = request.get_json()
            prompt_text = user_input["prompt"]

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500
    

    try:
        response = chain.invoke({"input": prompt_text})
        print(response)
        return jsonify({"response": response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
