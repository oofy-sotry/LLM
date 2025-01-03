from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
from langchain import PromptTemplate
import os

# 0. HuggingFace Hub 로그인 (환경 변수로 토큰 가져오기)
# token = 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
login(token=os.getenv("HUGGINGFACE_TOKEN"))
print("----------------------------------------------------------------------------------------------------")

# 1. 데이터 로드(Load Data) - 웹 문서 사용
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)
try:
    docs = loader.load()
    if not docs:
        raise ValueError("로드된 문서가 비어 있습니다.")
    print(f"문서 개수: {len(docs)}, 첫 번째 문서 길이: {len(docs[0].page_content)}")
    print(docs[0].page_content[5000:6000])
except Exception as e:
    print(f"데이터 로드 중 오류 발생: {e}")
    docs = []

print("----------------------------------------------------------------------------------------------------")

# 2. 텍스트 분할(Text Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
splits = text_splitter.split_documents(docs)
print(f"분할된 텍스트 개수: {len(splits)}, 예시 분할 텍스트: {splits[10]}")

print("----------------------------------------------------------------------------------------------------")

# 3. 인덱싱(Indexing) : 텍스트 -> 임베딩 -> 저장
# 임베딩 : 문장을 숫자로 변환하는 것
embeddings_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-nli",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("----------------------------------------------------------------------------------------------------")

# 4. 임베딩 계산
try:
    embeddings = embeddings_model.embed_documents([split.page_content for split in splits])
    print(f"임베딩 개수: {len(embeddings)}, 첫 번째 임베딩 길이: {len(embeddings[0])}")
except Exception as e:
    print(f"임베딩 계산 중 오류 발생: {e}")

print("----------------------------------------------------------------------------------------------------")

# 5. Vector Store : FAISS 사용 - CPU 사용 버전
text_embeddings = list(zip([split.page_content for split in splits], embeddings))
vectorstore = FAISS.from_embeddings(
    text_embeddings=text_embeddings,  
    embedding=embeddings_model,   
    distance_strategy=DistanceStrategy.COSINE
)

print("----------------------------------------------------------------------------------------------------")

# 6. Vector Store 저장
try:
    vectorstore.save_local('./db/faiss')
except Exception as e:
    print(f"Vector Store 저장 중 오류 발생: {e}")

print("----------------------------------------------------------------------------------------------------")

# 7. 검색
query = "최상위 정책은 뭐야??"
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 3, 'fetch_k': 10}
)

try:
    docs = retriever.get_relevant_documents(query)
    print(f"검색 결과 문서 개수: {len(docs)}, 첫 번째 문서 내용: {docs[0].page_content[:500]}")

    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    
    context = ""
    for doc in docs:
        content = doc.page_content.strip()
        result = qa_pipeline(question=query, context=content)
        context += result['answer'] + "\n\n"

except Exception as e:
    print(f"검색 중 오류 발생: {e}")
    context = "검색된 내용이 없습니다."

print("----------------------------------------------------------------------------------------------------")

# 8. prompt 설정
template = '''
다음의 형식과 내용으로 질문에 답변하세요.
사용자가 한 질문을 먼저 보여주고, 그에 대한 답변을 보여주세요.
마지막으로 그 답변에 대한 내용을 찾은 부분을 보여주세요.
'''

# 포맷 함수
def format_docs(docs, max_length=4096):
    total_length = 0
    selected_docs = []

    for doc in docs:
        doc_length = len(doc.page_content)
        if total_length + doc_length > max_length:
            break
        selected_docs.append(doc.page_content)
        total_length += doc_length

    return '\n\n'.join(selected_docs)

print("----------------------------------------------------------------------------------------------------")

# 9. Peft 및 토크나이저 모델 로드
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

print("----------------------------------------------------------------------------------------------------")

# 10. 텍스트 추론 및 결과 생성
input_text = f"""
다음은 여러 문서에서 추출한 내용입니다. 이 내용을 바탕으로 질문에 대한 답변을 생성하세요.

문서 내용:
{context}

질문: {query}
답변:
""" 

inputs = tokenizer(input_text, return_tensors='pt', max_length=4096, truncation=True)

try:
    generated_text = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
    final_output = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print(f"질문 및 답변 : {final_output}")
except Exception as e:
    print(f"텍스트 생성 중 오류 발생: {e}")

print("----------------------------------------------------------------------------------------------------")