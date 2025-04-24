import faiss
import numpy as np
import streamlit as st
import logging
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Налаштування логування
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # Лог у файл
        logging.StreamHandler()          # І в консоль
    ]
)

@st.cache_resource
def load_local_model(model_name):
    logging.info("Завантаження локальної моделі...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token  # Використовуємо `eos_token` як `pad_token`
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Помилка при завантаженні моделі: {e}")
        st.error("Не вдалося завантажити модель. Перевірте назву або підключення до Інтернету.")
        raise

@st.cache_resource
def initialize_faiss(documents, _embedding_model):
    logging.info("Ініціалізація FAISS...")
    try:
        document_embeddings = _embedding_model.encode(
            documents, show_progress_bar=True, normalize_embeddings=True
        )
        logging.info(f"Отримано вектори для документів: {document_embeddings.shape}")
        dimension = document_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(document_embeddings, dtype=np.float32))
        logging.info(f"Документи додано в FAISS: {index.ntotal} елементів")
        return index, document_embeddings
    except Exception as e:
        logging.error(f"Помилка при ініціалізації FAISS: {e}")
        st.error("Не вдалося ініціалізувати FAISS. Перевірте правильність даних.")
        raise

def search_documents(query, index, _embedding_model, documents, k=3):
    logging.info(f"Шукаємо документи для запиту: {query}")
    try:
        query_embedding = _embedding_model.encode([query], normalize_embeddings=True)
        _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=k)
        unique_indices = list(set(indices[0]))  # Унікальні індекси
        logging.info(f"Знайдено унікальні документи: {unique_indices}")
        return [documents[i] for i in unique_indices]
    except Exception as e:
        logging.error(f"Помилка при пошуку документів: {e}")
        st.error("Сталася помилка при пошуку документів. Спробуйте знову.")
        raise

def truncate_documents(documents, max_length=300):
    logging.info(f"Обрізаємо документи до {max_length} символів.")
    return [doc[:max_length] for doc in documents]

def generate_answer(prompt, model, tokenizer):
    logging.info(f"Генерація відповіді для запиту: {prompt}")
    try:
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
        logging.info(f"Генерована відповідь: {answer}")
        return answer
    except Exception as e:
        logging.error(f"Помилка при генерації відповіді: {e}")
        st.error("Не вдалося згенерувати відповідь. Спробуйте знову.")
        raise

def rag_system(query, index, documents, model, tokenizer, _embedding_model):
    logging.info(f"Запит до RAG-системи: {query}")
    try:
        relevant_documents = search_documents(query, index, _embedding_model, documents)
        truncated_documents = truncate_documents(relevant_documents)
        prompt = (
            f"Знайдені документи:\n" 
            f"{' '.join(truncated_documents)}\n" 
            f"Запит: {query}\nВідповідь:"
        )
        answer = generate_answer(prompt, model, tokenizer)
        return answer
    except Exception as e:
        logging.error(f"Помилка в RAG-системі: {e}")
        st.error("Сталася помилка в RAG-системі. Спробуйте знову.")
        raise

def run_interface():
    logging.info("Запуск інтерфейсу RAG-системи")
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

    logging.info("Завершення роботи інтерфейсу")

if __name__ == "__main__":
    logging.info("Запуск програми RAG-системи")
    run_interface()
    logging.info("Завершення роботи програми")
