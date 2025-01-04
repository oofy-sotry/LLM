from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import torch

def perform_search(query):
    url = 'https://ko.wikipedia.org/wiki/%EC%9C%84%ED%82%A4%EB%B0%B1%EA%B3%BC:%EC%A0%95%EC%B1%85%EA%B3%BC_%EC%A7%80%EC%B9%A8'
    try:
        # 데이터 로드
        loader = WebBaseLoader(url)
        docs = loader.load()
        print("데이터 로드 완료")
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print("텍스트 분할 완료")

    # 임베딩 계산
    embeddings_model = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-nli",
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    embeddings = embeddings_model.embed_documents([split.page_content for split in splits])
    print("임베딩 완료")

    # Vector Store 생성
    text_embeddings = list(zip([split.page_content for split in splits], embeddings))
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings_model,
        distance_strategy=DistanceStrategy.COSINE
    )
    print("벡터 저장소 생성 완료")

    # 쿼리로 검색
    retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'fetch_k': 10})
    docs = retriever.get_relevant_documents(query)
    print("검색 완료")

    # 검색 결과를 JSON 직렬화 가능 형식으로 변환
    results = [
        {
            "page_content": doc.page_content,  # 문서 내용
            "metadata": doc.metadata  # 문서 메타데이터 (필요한 경우)
        }
        for doc in docs
    ]
    print(f"검색 결과: {results}")
    return results
