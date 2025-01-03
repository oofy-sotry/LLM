from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForQuestionAnswering, pipeline
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
    search_kwargs={'k': 3, 'fetch_k': 10}
)

try:
    # 관련 문서 검색
    docs = retriever.get_relevant_documents(query)
    print(f"검색 결과 문서 개수: {len(docs)}, 첫 번째 문서 내용: {docs[0]}")

    # 검색된 문서들을 바탕으로 context 만들기
    seen = set()
    context = ""
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            context += content + "\n\n"

    print("----------------------------------------------------------------------------------------------------")

    # 8. QA 모델로 질문에 대한 답변 추출
    answertokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    answermodel = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

    # QA 모델 파이프라인을 사용하여 답변 추출
    qa_pipeline = pipeline("question-answering", model=answermodel, tokenizer=answertokenizer)

    # 검색된 문서들을 바탕으로 context에서 질문에 대한 답을 찾기
    result = qa_pipeline(question=query, context=context)
    qa_answer = result['answer']
    print(f"QA 모델로 추출한 답변: {qa_answer}")

    # 9. LLM을 사용하여 자연스러운 답변 생성
    # LLM 모델 로드
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

    # 입력 텍스트 구성
    input_text = f"""
    질문: {query}
    QA 모델로 추출한 답변: {qa_answer}
    """

    inputs = tokenizer(input_text, return_tensors='pt', max_length=4096, truncation=True)

    # LLM을 통해 최종 답변 생성
    try:
        generated_text = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7
        )
        final_output = tokenizer.decode(generated_text[0], skip_special_tokens=True)
        print(f"최종 생성된 답변: {final_output}")
    except Exception as e:
        print(f"텍스트 생성 중 오류 발생: {e}")

except Exception as e:
    print(f"검색 중 오류 발생: {e}")
    context = "검색된 내용이 없습니다."

print("----------------------------------------------------------------------------------------------------")
