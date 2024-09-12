from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import HuggingFacePipeline
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

app = Flask(__name__)
CORS(app)

# 모델과 토크나이저 로드
config = PeftConfig.from_pretrained("jeunghyen/llama-2-ko-7b-1")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "jeunghyen/llama-2-ko-7b-1")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# HuggingFace pipeline 생성 (LangChain에서 사용하기 위함)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# LangChain의 HuggingFace LLM 객체 생성
llm = HuggingFacePipeline(pipeline=pipe)

# Prompt 템플릿 생성
template = "You are an expert in astronomy. Answer the question.\nQuestion: {input}"
prompt = PromptTemplate(input_variables=["input"], template=template)

# LLMChain 생성 (PromptTemplate과 모델을 연결)
chain = LLMChain(llm=llm, prompt=prompt)

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.get_json()
        if user_input:
            prompt_text = user_input["prompt"]
        else:
            prompt_text = "데이터가 들어오지 않아 질문이 없습니다."

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        # LLMChain을 통해 모델 응답 생성
        response = chain.run(input=prompt_text)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
