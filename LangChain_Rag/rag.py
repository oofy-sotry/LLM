# 답변을 생성하는 속도가 너무 느림
# 수정한 부분
# 1. 입력 텍스트 길이 조정
#   - text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
# 2. top_k, top_p 설정 조정
#   - 장점 : 시간을 줄일수 있음
#   - 단점 : 답변의 다양성 줄어듬
#   - output = model.generate(**inputs, max_new_tokens=512, top_k=20, top_p=0.9)
# 2.1 num_beams 값 조정(사용안함)
#   - 탐색 범위를 줄여서 시간을 줄임
#   - output = model.generate(**inputs, max_new_tokens=512, num_beams=1)
# 3. FP16 (반정밀도 연산) 사용
#   - base_model = AutoModelForCausalLM.from_pretrained(
#       "meta-llama/Llama-2-7b-hf",
#       torch_dtype=torch.float16
#     )
# 4. 문서 개수 및 검색 최적화
#   - retriever = vectorstore.as_retriever(
#       search_type='mmr',
#       search_kwargs={'k': 3, 'fetch_k': 30}
#     )

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login



login(token="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")
print("----------------------------------------------------------------------------------------------------")

# 1. 데이터 로드(Load Data) - 웹 문서 사용, 텍스트문서나 CSV문서 등 다른 방법도 가능
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)
docs = loader.load()

print(len(docs))
print(len(docs[0].page_content))
print(docs[0].page_content[5000:6000])

print("----------------------------------------------------------------------------------------------------")

# 2. 텍스트 분할(Text Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

splits = text_splitter.split_documents(docs)

print(len(splits))
print(splits[10])

print("----------------------------------------------------------------------------------------------------")

# 3. 인덱싱(Indexing) : 텍스트 -> 임베딩 -> 저장
# 수정 이유 : name 이라는 변수를 이제는 사용하지 않음, model_name이라는 변수로 수정
embeddings_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-nli",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

print("----------------------------------------------------------------------------------------------------")

# 4. 임베딩 계산
embeddings = embeddings_model.embed_documents([split.page_content for split in splits])
print(f"임베딩 개수: {len(embeddings)}, 첫 번째 임베딩 길이: {len(embeddings[0])}")
print("----------------------------------------------------------------------------------------------------")

# 5. Vector Store : FAISS 사용 - CPU 사용 버전 사용
# 수정 1 - 이유 : FAISS 벡터스토어는 임베딩 벡터와 해당하는 문서를 함께 받아야 함
# 수정 2 - 이유 : text_embeddings는 텍스트와 해당 텍스트에 대한 임베딩을 짝지은 튜플이어야 함

text_embeddings = list(zip([split.page_content for split in splits], embeddings))

vectorstore = FAISS.from_embeddings(
    text_embeddings=text_embeddings,  # 기존의 embeddings 리스트 사용
    embedding=embeddings_model,   # 문서의 임베딩을 생성한 모델 표시
    distance_strategy=DistanceStrategy.COSINE
)

print("----------------------------------------------------------------------------------------------------")

# 6. Vector Store 저장
vectorstore.save_local('./db/faiss')

print("----------------------------------------------------------------------------------------------------")

# 7. 검색
query = "위키백과의 정책에 대해서 알려줘"
# MMR - 다양성 고려 (lambda_mult = 0.5)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 3, 'fetch_k': 30}
)

docs = retriever.get_relevant_documents(query)
print(len(docs))
print(docs[0])

print("----------------------------------------------------------------------------------------------------")

# 8. prompt 설정
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

# 포맷 함수
# 수정 : 입력 텍스트 길이 조정
# 이유 : 입력 텍스트가 너무 길면 미리 잘라서 모델이 처리할 수 있는 범위로 조정하기 위해
def format_docs(docs, max_length=4096):
    """
    입력 텍스트의 길이가 너무 길 경우 모델에서 허용하는 길이로 잘라서 반환하는 함수
    max_length는 모델에서 허용하는 최대 입력 토큰 길이
    """
    total_length = 0
    selected_docs = []

    for doc in docs:
        doc_length = len(doc.page_content)
        if total_length + doc_length > max_length:
            break
        selected_docs.append(doc.page_content)
        total_length += doc_length

    return '\n\n'.join(selected_docs)

# 9. Peft 및 토크나이저 모델 로드
config = PeftConfig.from_pretrained("jeunghyen/llama-2-ko-7b-4")
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16
)
model = PeftModel.from_pretrained(base_model, "jeunghyen/llama-2-ko-7b-4")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

print("----------------------------------------------------------------------------------------------------")

# 10. 텍스트 추론 및 결과 생성
# max_length로 입력 텍스트 길이를 조정
input_text = template.format(context=format_docs(docs, max_length=4096), question=query)
inputs = tokenizer(input_text, return_tensors='pt')
# 수정 이유 : 답변에 대한 길이로 인한 오류 발생
output = model.generate(**inputs, max_new_tokens=512, top_k=20, top_p=0.9)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print("----------------------------------------------------------------------------------------------------")

# 11. 응답 출력
print(response)
