# 패키지 설치
!pip install langchain
!pip install faiss-gpu
!pip install peft
!pip install transformers
!pip install unstructured
!pip install -U langchain-community
!pip install langchain_huggingface
!pip install huggingface_hub

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

url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
loader = WebBaseLoader(url)
docs = loader.load()

print(len(docs))
print(len(docs[0].page_content))
print(docs[0].page_content[5000:6000])

print("----------------------------------------------------------------------------------------------------")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(len(splits))
print(splits[10])

print("----------------------------------------------------------------------------------------------------")

embeddings_model = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sroberta-nli",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

print("----------------------------------------------------------------------------------------------------")
embeddings = embeddings_model.embed_documents([split.page_content for split in splits])
print(f"임베딩 개수: {len(embeddings)}, 첫 번째 임베딩 길이: {len(embeddings[0])}")

print("----------------------------------------------------------------------------------------------------")

text_embeddings = list(zip([split.page_content for split in splits], embeddings))

vectorstore = FAISS.from_embeddings(
    text_embeddings=text_embeddings,
    embedding=embeddings_model,
    distance_strategy=DistanceStrategy.COSINE
)

print("----------------------------------------------------------------------------------------------------")

vectorstore.save_local('./db/faiss')

print("----------------------------------------------------------------------------------------------------")

query = "위키백과의 정책에 대해서 알려줘"
retriever = vectorstore.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 5, 'fetch_k': 50}
)

docs = retriever.get_relevant_documents(query)
print(len(docs))
print(docs[0])

print("----------------------------------------------------------------------------------------------------")

template = '''Answer the question based only on the following context:
{context}

Question: {question}
'''

print("----------------------------------------------------------------------------------------------------")

def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])

print("----------------------------------------------------------------------------------------------------")

config = PeftConfig.from_pretrained("jeunghyen/llama-2-ko-7b-4")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, "jeunghyen/llama-2-ko-7b-4")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

print("----------------------------------------------------------------------------------------------------")

input_text = template.format(context=format_docs(docs), question=query)
inputs = tokenizer(input_text, return_tensors='pt')
# 수정 이유 : 답변에 대한 길이로 인한 오류 발생
output = model.generate(**inputs, max_new_tokens=512)
response = tokenizer.decode(output[0], skip_special_tokens=True)

print("----------------------------------------------------------------------------------------------------")
print(response)


