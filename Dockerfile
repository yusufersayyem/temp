FROM python:3.11-slim

WORKDIR /app

# نسخ ملف متطلبات المكتبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي ملفات المشروع
COPY . .

# المنفذ المفتوح للتطبيق
EXPOSE 8000

# أمر تشغيل التطبيق على المنفذ المناسب لـ Render
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]