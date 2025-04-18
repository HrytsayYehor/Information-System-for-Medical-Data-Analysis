# 🧠 RAG-система для медичних даних

Цей проєкт — демонстрація локальної RAG (Retrieval-Augmented Generation) системи для медичних запитів. Система використовує:
- Векторний пошук (FAISS)
- Генеративну модель (`GPT-Neo 2.7B`)
- Streamlit для інтерактивного інтерфейсу

---

## 🚀 Можливості

- Пошук релевантних медичних записів за запитом
- Генерація відповідей на базі знайдених документів
- Повністю локальне виконання без потреби в API

---

## 🧰 Технології

- `Python`
- `transformers`, `sentence-transformers`
- `FAISS` (векторна база даних)
- `Streamlit`
- `Torch`
- Модель: [`EleutherAI/gpt-neo-2.7B`](https://huggingface.co/EleutherAI/gpt-neo-2.7B)

---

## ⚙️ Як запустити

### 1. Клонуй репозиторій

```bash
git clone https://github.com/твій-користувач/RAG-Medical.git
cd RAG-Medical
```
  

### Створи віртуальне середовище
``````
python -m venv .venv
source .venv/bin/activate    # або .venv\Scripts\activate у Windows
``````


### Структура проєкту
````
DiplomFinal0/
├── .venv/                  ← Віртуальне середовище (ігнорується у Git)
├── rag.py                  ← Основний скрипт RAG-системи
├── RAGF.py                 ← Альтернативний або допоміжний скрипт
├── test_rag.py             ← Тести або експериментальні виклики
├── docx-template/          ← Шаблони або документи
├── debug.log               ← Логи (ігноруються у Git)
├── pyvenv.cfg              ← Конфігурація вірт. середовища (ігнорується)
└── README.md               ← Поточний файл документації
