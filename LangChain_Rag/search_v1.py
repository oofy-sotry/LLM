from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from huggingface_hub import login
import torch
import os
import requests
import json

# 0. HuggingFace Hub 로그인 (환경 변수로 토큰 가져오기)
# token = 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
login(token=os.getenv("HUGGINGFACE_TOKEN"))
print("----------------------------------------------------------------------------------------------------")
print("1번")
# 1. 데이터 로드(Load Data) - 웹 문서 사용
url1 = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
url2 = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%ED%8E%B8%EC%A7%91_%EC%A7%80%EC%B9%A8'

loader = WebBaseLoader(
    web_paths=(url1, url2)
)
try:
    docs = loader.load()
    if not docs:
        raise ValueError("로드된 문서가 비어 있습니다.")
    # print(f"문서 개수: {len(docs)}, 첫 번째 문서 길이: {len(docs[0].page_content)}")
    # print(docs[0].page_content[5000:6000])
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    docs =  []

print("----------------------------------------------------------------------------------------------------")
print("2번")
# 2. 텍스트 분할(Text Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
# print(f"분할된 텍스트 개수: {len(splits)}, 예시 분할 텍스트: {splits[10]}")

print("----------------------------------------------------------------------------------------------------")
print("3번")
# 3. 인덱싱(Indexing) : 텍스트 -> 임베딩 -> 저장
# 임베딩 : 문장을 숫자로 변환하는 것
embeddings_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-nli",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("----------------------------------------------------------------------------------------------------")
print("4번")
# 4. 임베딩 계산
try:
    embeddings = embeddings_model.embed_documents([split.page_content for split in splits])
    # print(f"임베딩 개수: {len(embeddings)}, 첫 번째 임베딩 길이: {len(embeddings[0])}")
except Exception as e:
    print(f"임베딩 계산 중 오류 발생: {e}")

print("----------------------------------------------------------------------------------------------------")
print("5번")
# 5. Vector Store : FAISS 사용 - CPU 사용 버전
text_embeddings = list(zip([split.page_content for split in splits], embeddings))
vectorstore = FAISS.from_embeddings(
    text_embeddings=text_embeddings,  
    embedding=embeddings_model,   
    distance_strategy=DistanceStrategy.COSINE
)

print("----------------------------------------------------------------------------------------------------")
print("6번")
# 6. Vector Store 저장
try:
    vectorstore.save_local('./db/faiss')
except Exception as e:
    print(f"Vector Store 저장 중 오류 발생: {e}")

print("----------------------------------------------------------------------------------------------------")
print("7번")
# 7. 검색
query = "최고 지침은 뭐야"
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 3, 'fetch_k': 30}
)
try:
    # 관련 문서 검색
    docs = retriever.get_relevant_documents(query)
    # print(f"검색 결과 문서 개수: {len(docs)}, 첫 번째 문서 내용: {docs[0]}")
except Exception as e:
    print(f"검색 중 오류 발생: {e}")
    context = "검색된 내용이 없습니다."
print("----------------------------------------------------------------------------------------------------")
print("8번")
# 최종 검색 내용
print("최종 검색 내용")
print(docs)

try:
    page_content = [ doc.page_content for doc in docs]
    print("최종 검색 내용중 필요 요소")
    for content in page_content:
        print(content)
except Exception as e:
    print(f"page_content 추충 중 오류 발생: {e}")

print("----------------------------------------------------------------------------------------------------")
print("9번")
# 9. API로 query와 page_content 전송
api_url = "http://127.0.0.1:5000/process"
headers = {"Content-Type": "application/json"}

try:
    # 데이터를 JSON 형식으로 준비
    payload = {"query": query, "contents": content}
    response = requests.post(api_url, headers=headers, data=json.dumps(payload))

    # 응답 처리
    if response.status_code == 200:
        print("데이터가 성공적으로 전송되었습니다.")
        print("API 응답:", response.json())
    else:
        print(f"API 요청 실패: {response.status_code}, {response.text}")
except Exception as e:
    print(f"API 요청 중 오류 발생: {e}")