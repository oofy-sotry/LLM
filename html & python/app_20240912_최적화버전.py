from flask import Flask, request, jsonify
from flask_cors import CORS
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

# 모델과 토크나이저 로드
config = PeftConfig.from_pretrained("jeunghyen/llama-2-ko-7b-1")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "jeunghyen/llama-2-ko-7b-1")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_input = request.get_json()
        prompt_text = user_input["prompt"] if user_input else "데이터가 들어오지 않아 질문이 없습니다."

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    try:
        # 모델 입력 토크나이즈
        inputs = tokenizer(prompt_text, return_tensors="pt")
        # 모델로부터 응답 생성
        outputs = model.generate(**inputs)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"response": response})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
