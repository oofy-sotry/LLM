from flask import Flask, request, jsonify
from search import perform_search
from llm import generate_answer_from_llm
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS 활성화

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the API!"})

@app.route('/favicon.ico')
def favicon():
    return '', 204

# 1단계: 쿼리와 내용을 저장 (POST)
@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        if not isinstance(data, dict) or "query" not in data:
            return jsonify({"error": "'query' field is required in JSON"}), 400
        
        query = data.get('query', '')
        search_results = perform_search(query)  # JSON 직렬화 가능한 데이터로 반환
        print(f"Search results: {search_results}")
        
        return jsonify({"query": query, "contents": search_results}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

# 2단계: LLM을 이용해 최종 답변 생성 (POST)
@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    try:
        data = request.get_json()
        query = data.get('query', '')
        contents = data.get('contents', [])
        print(f"Received for LLM: query={query}, contents={contents}")

        # LLM 모델을 통해 최종 답변 생성
        answer = generate_answer_from_llm(query, contents)
        print(f"Generated answer: {answer}")

        return jsonify({"response": answer}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Flask 서버는 포트 5000에서 실행
