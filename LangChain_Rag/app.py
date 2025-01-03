from flask import Flask, request, jsonify
import json
from search import perform_search
from llm import generate_answer_from_llm
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS 활성화

# 쿼리와 내용을 저장할 변수
data_store = {"query": "", "contents": []}

# 루트 URL 처리
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the API!"})

# Favicon 요청 처리
@app.route('/favicon.ico')
def favicon():
    return '', 204

# 1단계: 쿼리와 내용을 저장 (POST)
@app.route('/process', methods=['POST'])
def process_data():
    try:
        print("Raw request data:", request.data)
        print("Request content type:", request.content_type)

        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        data = request.get_json()
        print("Parsed JSON data:", data)

        if not isinstance(data, dict) or "query" not in data:
            return jsonify({"error": "'query' field is required in JSON"}), 400

        query = data.get('query', '')
        print("query : " + query)

        # 검색 수행
        search_results = perform_search(query)
        data_store['query'] = query
        data_store['contents'] = search_results

        print("contents : ")
        print(search_results)
        
        return jsonify({
            "message": "데이터가 성공적으로 저장되었습니다.",
            "query": query,
            "contents": search_results,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# 2단계: LLM을 이용해 최종 답변 생성 (POST)
@app.route('/generate_answer', methods=['POST'])
def generate_answer():
    print("llm 생성 함수 시작")
    try:
        data = request.get_json()
        query = data.get('query', '')
        contents = data.get('contents', [])
        
        # LLM 모델을 통해 최종 답변 생성 (llm.py에서 구현된 generate_answer_from_llm 함수 호출)
        answer = generate_answer_from_llm(query, contents)

        return jsonify({"response": answer}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)  # Flask 서버는 포트 5000에서 실행
