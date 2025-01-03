from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import PromptTemplate
import torch

def generate_answer_from_llm(query, contents):
    try:
    # LLM 모델 로드
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    except Exception as e:
        print(f"Error loading model: {e}")
        return "LLM 모델 로드 오류"

    prompt_template = PromptTemplate(
        input_variables=["query", "content"],
        template="질문: {query}\n답변: {content}"
    )

    content = "\n".join(contents)
    prompt = prompt_template.format(query=query, content=content)

    inputs = tokenizer(prompt, return_tensors='pt', max_length=4096, truncation=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # LLM을 통해 최종 답변 생성
    generated_text = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.7,
        top_p=0.95,
        top_k=50,
        no_repeat_ngram_size=2
    )

    decoded_output = tokenizer.decode(generated_text[0], skip_special_tokens=True)
    return decoded_output
