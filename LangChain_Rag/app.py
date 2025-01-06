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

# 1단계: 검색 요청 처리
@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        print(f"Received data: {data}")
        
        if not data or "query" not in data:
            return jsonify({"error": "'query' field is required in JSON"}), 400
        
        query = data.get('query', '')
        search_results = perform_search(query)  # JSON 직렬화 가능한 데이터 반환
        print("Search results:", search_results)

        if not search_results:
            return jsonify({"query": query, "contents": []}), 200

        return jsonify({"query": query, "contents": search_results}), 200
    except Exception as e:
        print(f"Error in /process: {e}")
        return jsonify({"error": str(e)}), 500

# 2단계: LLM 요청 처리
@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    try:
        data = request.get_json()
        print(f"Received for LLM: {data}")

        query = data.get('query', '')
        print(query)
        pageContents = data.get('pageContents', [])
        print(pageContents)
        
        if not query or not pageContents:
            return jsonify({"error": "'query' and 'contents' fields are required"}), 400

        answer = generate_answer_from_llm(query, pageContents)
        print(f"Generated answer: {answer}")

        return jsonify({"response": answer}), 200
    except Exception as e:
        print(f"Error in /generate_answer: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Flask 서버 실행