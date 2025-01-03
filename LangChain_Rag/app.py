from flask import Flask, request, jsonify

app = Flask(__name__)
data_store = {"query": "", "contents": []}

# 루트 URL 처리
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the API!"})

# Favicon 요청 처리
@app.route('/favicon.ico')
def favicon():
    return '', 204

# 데이터 전송 (POST)
@app.route('/process', methods=['POST'])
def process_data():
    try:
        data = request.get_json()
        data_store['query'] = data.get('query', '')
        data_store['contents'] = data.get('contents', [])
        return jsonify({"message": "데이터가 성공적으로 저장되었습니다."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 데이터 조회 (GET)
@app.route('/process', methods=['GET'])
def get_data():
    try:
        return jsonify(data_store), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
