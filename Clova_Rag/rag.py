import json
import os
import subprocess
from langchain_community.document_loaders import UnstructuredHTMLLoader
from pathlib import Path
import base64
import http.client
from tqdm import tqdm
import requests

# 1. Raw Data -> Connecting

## 1.1 txt -> html 변환 및 원본 사이트 주소 mapping
url_to_filename_map = {}

with open("url.txt", "r") as file:
    urls = [url.strip() for url in file.readlines()]

folder_path = "repository"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for url in urls:
    filename = url.split("/")[-1] + ".html"
    file_path = os.path.join(folder_path, filename)
 #   subprocess.run(["wget", "-O", file_path, url], check=True)
    subprocess.run(
        [
            "wget",
            "--user-agent='MyCrawler/1.0 (+http://mycrawler.com/info)'",
            "-O",
            file_path,
            url,
        ],
        check=True,
    )
    url_to_filename_map[url] = filename

with open("url_to_filename_map.json", "w") as map_file:
    json.dump(url_to_filename_map, map_file)


# 1.2 LangChain 활용 HTML 로딩
html_files_dir = Path('/home/oofy/Documents/RAG/repository')

html_files = list(html_files_dir.glob("*.html"))

clovastudiodatas = []

for html_file in html_files:
    loader = UnstructuredHTMLLoader(str(html_file))
    document_data = loader.load()
    clovastudiodatas.append(document_data)
    print(f"Processed {html_file}")


# 1.3 Mapping 정보를 활용해 source를 실제 URL로 대체
with open("url_to_filename_map.json", "r") as map_file:
    url_to_filename_map = json.load(map_file)

filename_to_url_map = {v: k for k, v in url_to_filename_map.items()}

# clovastudiodatas 리스트의 각 Document 객체의 'source' 수정
for doc_list in clovastudiodatas:
    for doc in doc_list:
        extracted_filename = doc.metadata["source"].split("/")[-1]
        if extracted_filename in filename_to_url_map:
            doc.metadata["source"] = filename_to_url_map[extracted_filename]
        else:
            print(f"Warning: {extracted_filename}에 해당하는 URL을 찾을 수 없습니다.")


# 2. Chunking