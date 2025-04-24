
# Інструкція з розгортання у production-середовищі

Ця інструкція призначена для DevOps/Release Engineer, які розгортають проєкт **Information System for Medical Data Analysis** у production.

---

## 🖥️ Вимоги до апаратного забезпечення

- **Архітектура:** x86_64
- **CPU:** 4 ядра (рекомендовано 8)
- **RAM:** мінімум 8 GB (рекомендовано 16 GB)
- **Диск:** SSD, 10+ GB вільного місця
- **GPU:** (опціонально) для пришвидшення GPT-Neo – 1 GPU з 10+ GB VRAM (наприклад, NVIDIA RTX 3080)

---

## 💿 Необхідне програмне забезпечення

- **Ubuntu 20.04+** (або інша Linux-сумісна ОС)
- **Python 3.10+**
- **pip**
- **Git**
- **virtualenv**
- **Docker** *(опціонально)*
- **Nginx** *(для реверс-проксі)*

---

## 🌐 Налаштування мережі

- Відкритий порт: `80` або `443` (якщо використовується HTTPS)
- Доступ до інтернету для завантаження моделей
- (Опціонально) обмеження доступу по IP / VPN

---

## ⚙️ Конфігурація серверів

1. Створіть системного користувача `medapp`
2. Встановіть необхідні пакети:

```bash
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git nginx
```

3. Створіть директорію для додатку:

```bash
mkdir -p /opt/medapp && cd /opt/medapp
```

---

## 🛢️ Налаштування СУБД

> У цьому проєкті **база даних не використовується**. Всі дані обробляються локально у памʼяті.

---

## 📦 Розгортання коду

```bash
git clone https://github.com/HrytsayYehor/Information-System-for-Medical-Data-Analysis.git
cd Information-System-for-Medical-Data-Analysis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> Переконайтесь, що PyTorch встановлено з підтримкою GPU, якщо планується запуск моделі на GPU.

---

## ✅ Перевірка працездатності

1. Запуск Streamlit:

```bash
streamlit run main.py --server.port 8501
```

2. Перевірте в браузері: `http://<IP_СЕРВЕРА>:8501`

3. Ви маєте побачити інтерфейс з полем для введення запиту.

4. Введіть запит (наприклад, "лікування при гіпертонії") і переконайтесь у наявності відповіді.

---

✅ Готово! Система розгорнута у production-середовищі.
