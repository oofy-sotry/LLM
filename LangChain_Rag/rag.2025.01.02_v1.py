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
from langchain.prompts import PromptTemplate


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
    docs =  []

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
    search_kwargs={'k': 3, 'fetch_k': 10}  # fetch_k 값을 최적화하여 조정
)

try:
    docs = retriever.get_relevant_documents(query)
    print(f"검색 결과 문서 개수: {len(docs)}, 첫 번째 문서 내용: {docs[0]}")
except Exception as e:
    print(f"검색 중 오류 발생: {e}")

print("----------------------------------------------------------------------------------------------------")

# 8. prompt 설정
template = '''
Answer the question based only on the following context:
{docs}

Question: {query}
'''

prompt = PromptTemplate(template=template)

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
input_text = template.prompt(
    context=format_docs(docs[:3], max_length=2048),
    question=query
)

inputs = tokenizer(input_text, return_tensors='pt', max_length=4096, truncation=True)

# 토큰을 배치로 생성하여 최적화
try:
    generated_text = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7
    )
    final_output = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    print(f"최종 생성 텍스트: {final_output}")
except Exception as e:
    print(f"텍스트 생성 중 오류 발생: {e}")

print("----------------------------------------------------------------------------------------------------")