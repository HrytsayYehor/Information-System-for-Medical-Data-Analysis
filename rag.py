import faiss
import numpy as np
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import streamlit as st

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Кешована функція для завантаження моделі
@st.cache_resource
def load_local_model(model_name):
    logger.info("Завантаження локальної моделі...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    return model, tokenizer

# Кешована функція для ініціалізації FAISS
@st.cache_resource
def initialize_faiss(documents, _embedding_model):
    logger.info("Ініціалізація FAISS...")
    assert len(documents) > 0, "Документи не повинні бути порожніми."
    document_embeddings = _embedding_model.encode(documents, show_progress_bar=True, normalize_embeddings=True)
    logger.info(f"Отримано вектори для документів: {document_embeddings.shape}")
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(document_embeddings, dtype=np.float32))
    logger.info(f"Документи додано в FAISS: {index.ntotal} елементів")
    return index, document_embeddings

# Функція для пошуку документів
def search_documents(query, index, _embedding_model, documents, k=3):
    logger.info(f"Шукаємо документи для запиту: {query}")
    try:
        query_embedding = _embedding_model.encode([query], normalize_embeddings=True)
        _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=k)
        unique_indices = list(set(indices[0]))
        logger.info(f"Знайдено унікальні документи: {unique_indices}")
        return [documents[i] for i in unique_indices]
    except Exception as e:
        logger.error(f"Помилка при пошуку документів: {e}")
        return []

# Функція для скорочення документів
def truncate_documents(documents, max_length=300):
    return [doc[:max_length] for doc in documents]

# Функція для генерації відповіді
def generate_answer(prompt, model, tokenizer):
    logger.info(f"Генерація відповіді для запиту: {prompt}")
    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = inputs.to(model.device)
        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=50,
            temperature=0.5,
            top_p=0.9,
            top_k=30,
            repetition_penalty=1.2,
            length_penalty=1.0,
            do_sample=True
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Генерована відповідь: {answer}")
        return answer
    except Exception as e:
        logger.error(f"Помилка генерації відповіді: {e}")
        return "На жаль, не вдалося згенерувати відповідь."

# Функція для RAG-системи
def rag_system(query, index, documents, model, tokenizer, _embedding_model):
    logger.info(f"Запит до RAG-системи: {query}")
    relevant_documents = search_documents(query, index, _embedding_model, documents)
    if not relevant_documents:
        return "Не знайдено відповідних документів."
    truncated_documents = truncate_documents(relevant_documents)
    prompt = (
        f"Знайдені документи:\n"
        f"{' '.join(truncated_documents)}\n"
        f"Запит: {query}\nВідповідь:"
    )
    return generate_answer(prompt, model, tokenizer)

# Інтерфейс Streamlit
def run_interface():
    st.title("Локальна RAG-система для медичних даних")

    # Використовуємо GPT-Neo 2.7B як приклад
    model_name = "EleutherAI/gpt-neo-2.7B"
    model, tokenizer = load_local_model(model_name)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Документи для системи
    documents = [
        "Пацієнт P001 - Іванов Іван Іванович, 45 років. Діагноз: Гіпертонія. Лікування: Призначено антигіпертензивні препарати. Симптоми: головний біль, запаморочення.",
        "Пацієнт P002 - Петренко Петро Петрович, 60 років. Діагноз: Ішемічна хвороба серця. Лікування: Статини для зниження рівня холестерину, аспірин для профілактики тромбоутворення.",
        "Пацієнт P003 - Марченко Олена Олексіївна, 50 років. Діагноз: Гіпертонія. Лікування: Призначено діуретики, обмеження споживання солі, фізична активність.",
        "Пацієнт P004 - Литвиненко Сергій Володимирович, 55 років. Діагноз: Ішемічна хвороба серця. Лікування: Бета-блокатори, контроль стресу, регулярний моніторинг артеріального тиску.",
    ]

    documents = truncate_documents(documents)
    try:
        index, _ = initialize_faiss(documents, _embedding_model=embedding_model)
    except Exception as e:
        st.error(f"Помилка ініціалізації FAISS: {str(e)}")
        return

    query = st.text_input("Введіть запит:")

    if query.strip():
        answer = rag_system(query, index, documents, model, tokenizer, embedding_model)
        st.subheader("Відповідь:")
        st.write(answer)

if __name__ == "__main__":
    run_interface()
