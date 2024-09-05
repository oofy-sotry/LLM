from flask import Flask, request, jsonify
from langchain_community.llms import Ollama

app = Flask(__name__)

# Ollama 인스턴스 생성
llm = Ollama(model="llama2")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    prompt = data.get('prompt', '')

    try:
        response = llm(prompt)
        print(response)

        ai_response = response.get('text', '응답이 없습니다')
        return jsonify({'response': ai_response})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'API 호출 중 오류 발생'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
