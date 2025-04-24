
# Information System for Medical Data Analysis

Ласкаво просимо до проєкту! Це Streamlit-додаток, що реалізує локальну RAG-систему для медичних даних з використанням FAISS, SentenceTransformers та GPT-Neo.

## 🚀 Інструкція для розробника

Ця інструкція допоможе вам розгорнути проєкт на вашій машині з **чистою операційною системою**.

---

### ✅ Необхідні залежності та програмне забезпечення

1. **Python 3.10+**
2. **Git**
3. **pip**
4. **virtualenv** *(рекомендовано)*
5. **CUDA + PyTorch з GPU підтримкою** *(опційно, для пришвидшення GPT-Neo)*

---

### 🛠️ Клонування проєкту

```bash
git clone https://github.com/HrytsayYehor/Information-System-for-Medical-Data-Analysis.git
cd Information-System-for-Medical-Data-Analysis
```

---

### 🧪 Налаштування середовища розробки

> Рекомендується використання віртуального середовища:

```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

---

### 📦 Встановлення залежностей

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Якщо `requirements.txt` відсутній, створіть його з наступним вмістом:

```text
streamlit
torch
transformers
sentence-transformers
faiss-cpu
numpy
```

> Для використання GPU: замініть `faiss-cpu` на `faiss-gpu`, і встановіть `torch` з [офіційного сайту](https://pytorch.org/get-started/locally/).

---

### 🧠 Завантаження моделей (автоматично при запуску)

При першому запуску буде автоматично завантажено:
- **GPT-Neo 2.7B**: `EleutherAI/gpt-neo-2.7B`
- **SentenceTransformer**: `all-MiniLM-L6-v2`

Переконайтесь, що маєте стабільне інтернет-з’єднання.

---

### 🗄️ База даних

> ❗️ В даному проєкті **немає зовнішньої бази даних** — вектори та документи зберігаються в пам’яті у FAISS.

---

### 🚦 Запуск у режимі розробки

```bash
streamlit run main.py
```

або якщо ваш файл має іншу назву:

```bash
streamlit run your_filename.py
```

---

### 🔧 Базові операції

- У вікні браузера з’явиться інтерфейс.
- Введіть запит користувача (наприклад: `"Яке лікування при гіпертонії?"`)
- Додаток знайде релевантні документи та згенерує відповідь за допомогою GPT-Neo.

---

Готово! 🎉 Тепер ви можете експериментувати з RAG-системою для аналізу медичних даних.
