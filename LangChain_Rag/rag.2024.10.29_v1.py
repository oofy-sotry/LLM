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
input_text = template.format(context=format_docs(docs[:3]), question=query)  # 상위 3개의 문서만 사용
inputs = tokenizer(input_text, return_tensors='pt', max_length=4096, truncation=True)

# 토큰 생성 중간 상태를 추적하기 위해 반복적으로 토큰 생성
max_new_tokens = 256
generated_tokens = []

# 초기 입력으로 모델의 로짓 계산
with torch.no_grad():
    for step in range(max_new_tokens):
        # 토큰을 하나씩 생성
        output = model.generate(**inputs, max_new_tokens=1, do_sample=True)

        # 생성된 토큰을 추가
        new_token = output[:, -1].item()
        generated_tokens.append(new_token)

        # 현재까지 생성된 텍스트 출력 (옵션)
        current_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        print(f"Step {step + 1}/{max_new_tokens}: {current_output}")

        # 퍼센트 계산 및 출력
        percent_complete = (step + 1) / max_new_tokens * 100
        print(f"Progress: {percent_complete:.2f}%")

# 최종 생성된 텍스트 출력
final_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"Final Output: {final_output}")

print("----------------------------------------------------------------------------------------------------")
