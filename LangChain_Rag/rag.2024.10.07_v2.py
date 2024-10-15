from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.output_parsers import StrOutputParser



# 1. 데이터 로드(Load Data)
url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print(len(docs[0].page_content))
print(docs[0].page_content[5000:6000])

print("----------------------------------------------------------------------------------------------------")

# 2. 텍스트 분할(Text Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(len(splits))
print(splits[10])

print("----------------------------------------------------------------------------------------------------")

# 3. 인덱싱(Indexing) : 텍스트 -> 임베딩 -> 저장
#                       저장된 임베딩을 기반으로 유사성 검색을 수행하는 과정

model_name = "jhgan/ko-sroberta-nli"

embeddings_model = HuggingFaceEmbeddings(
    model=model_name,
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True},
)

# 4. 임베딩 계산
embeddings = embeddings_model.embed_documents([split.page_content for split in splits])
print(f"임베딩 개수: {len(embeddings)}, 첫 번째 임베딩 길이: {len(embeddings[0])}")

# 5. Vector Store : FAISS 사용 - CPU 사용 버전 사용
vectorstore = FAISS.from_embeddings(
    embeddings=embeddings,
    documents=splits,  # 분할된 문서를 사용합니다
    distance_strategy=DistanceStrategy.COSINE
)

# 6. Vector Store 저장
vectorstore.save_local('./db/faiss')

# 7. 검색
query = "위키백과의 정책에 대해서 알려줘"
# MMR - 다양성 고려 (lambda_mult = 0.5)
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

docs = retriever.get_relevant_documents(query)
print(len(docs))
docs[0]

# prompt 설정
template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

config = PeftConfig.from_pretrained("jeunghyen/llama-2-ko-7b-1")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "jeunghyen/llama-2-ko-7b-1")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

output_parser = StrOutputParser()

chain = prompt | model | output_parser

response = chain.invoke({'context': (format_docs(docs)), 'question':query})