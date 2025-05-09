import faiss
import numpy as np
import streamlit as st

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


@st.cache_resource
def load_local_model(model_name):
    print("Завантаження локальної моделі...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Використовуємо `eos_token` як `pad_token`
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    return model, tokenizer


@st.cache_resource
def initialize_faiss(documents, _embedding_model):
    print("Ініціалізація FAISS...")
    document_embeddings = _embedding_model.encode(
        documents, show_progress_bar=True, normalize_embeddings=True
    )
    print(f"Отримано вектори для документів: {document_embeddings.shape}")
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(document_embeddings, dtype=np.float32))
    print(f"Документи додано в FAISS: {index.ntotal} елементів")
    return index, document_embeddings


def search_documents(query, index, _embedding_model, documents, k=3):
    print(f"Шукаємо документи для запиту: {query}")
    query_embedding = _embedding_model.encode([query], normalize_embeddings=True)
    _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=k)
    unique_indices = list(set(indices[0]))  # Унікальні індекси
    print(f"Знайдено унікальні документи: {unique_indices}")
    return [documents[i] for i in unique_indices]


def truncate_documents(documents, max_length=300):
    return [doc[:max_length] for doc in documents]


def generate_answer(prompt, model, tokenizer):
    print(f"Генерація відповіді для запиту: {prompt}")
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True, truncation=True, max_length=256
    )  # Зменшення max_length для запобігання зависання
    inputs = inputs.to(model.device)
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        temperature=0.5,
        top_p=0.9,
        top_k=30,
        repetition_penalty=1.2,
        length_penalty=1.0,
        do_sample=True,
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Генерована відповідь: {answer}")
    return answer


def rag_system(query, index, documents, model, tokenizer, _embedding_model):
    print(f"Запит до RAG-системи: {query}")
    relevant_documents = search_documents(query, index, _embedding_model, documents)
    truncated_documents = truncate_documents(relevant_documents)
    prompt = (
        f"Знайдені документи:\n" f"{' '.join(truncated_documents)}\n" f"Запит: {query}\nВідповідь:"
    )
    answer = generate_answer(prompt, model, tokenizer)
    return answer


def run_interface():
    st.title("Локальна RAG-система для медичних даних")

    # Використовуємо GPT-Neo 2.7B як приклад
    model_name = "EleutherAI/gpt-neo-2.7B"  # Вибір моделі з 2.7B параметрів
    model, tokenizer = load_local_model(model_name)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Документи про високий тиск та серцево-судинні захворювання
    documents = [
        "Пацієнт P001 - Іванов Іван Іванович, 45 років. Діагноз: Гіпертонія. Лікування: Призначено антигіпертензивні препарати. Симптоми: головний біль, запаморочення.",
        "Пацієнт P002 - Петренко Петро Петрович, 60 років. Діагноз: Ішемічна хвороба серця. Лікування: Статини для зниження рівня холестерину, аспірин для профілактики тромбоутворення.",
        "Пацієнт P003 - Марченко Олена Олексіївна, 50 років. Діагноз: Гіпертонія. Лікування: Призначено діуретики, обмеження споживання солі, фізична активність.",
        "Пацієнт P004 - Литвиненко Сергій Володимирович, 55 років. Діагноз: Ішемічна хвороба серця. Лікування: Бета-блокатори, контроль стресу, регулярний моніторинг артеріального тиску.",
    ]

    documents = truncate_documents(documents)
    index, _ = initialize_faiss(documents, _embedding_model=embedding_model)

    query = st.text_input("Введіть запит:")

    if query:
        answer = rag_system(query, index, documents, model, tokenizer, embedding_model)
        st.subheader("Відповідь:")
        st.write(answer)


if __name__ == "__main__":
    run_interface()
