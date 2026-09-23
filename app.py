import json
import os
import random
import time
import requests
import chainlit as cl
import numpy as np

# 1. إعداد النموذج المفضّل والمستقر لـ Feature Extraction للغة العربية
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"

HF_TOKEN = os.environ.get("HF_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# متغيرات التخزين العامة
intents_mapping = []
patterns_embeddings = None


def get_embeddings_from_hf(texts, retries=3, delay=5):
    """دالة مساعدة لاستدعاء Hugging Face API وجلب المتجهات بشكل مستقر"""
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
            
            if response.status_code == 200:
                embeddings = np.array(response.json())
                
                # التأكد من الأبعاد 2D
                if len(embeddings.shape) == 1:
                    embeddings = np.expand_dims(embeddings, axis=0)

                # معايرة المتجهات (Normalization)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                return embeddings / norms
            else:
                print(f"محاولة {attempt + 1}: فشل الاستجابة ({response.status_code}) - {response.text}")
        except Exception as e:
            print(f"محاولة {attempt + 1}: خطأ في الاتصال ({e})")
            
        if attempt < retries - 1:
            time.sleep(delay)

    return None


def load_intents_data(json_path="intents.json"):
    """قراءة بيانات الملف وتحضير الأنماط"""
    global intents_mapping
    intents_mapping = []
    raw_patterns = []

    if not os.path.exists(json_path):
        print(f"خطأ: الملف {json_path} غير موجود!")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for intent in data.get("intents", []):
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            raw_patterns.append(pattern)
            intents_mapping.append({"tag": tag, "responses": responses})

    return raw_patterns


@cl.on_chat_start
async def start():
    """تنفذ هذه الدالة عند فتح المستخدم للمحادثة لأول مرة"""
    global patterns_embeddings
    
    # تحميل المتجهات إذا لم تكن محملة سابقاً
    if patterns_embeddings is None:
        raw_patterns = load_intents_data()
        if raw_patterns:
            msg = cl.Message(content="جاري إعداد وتحميل البيانات، يرجى الانتظار لحظات...")
            await msg.send()
            
            patterns_embeddings = get_embeddings_from_hf(raw_patterns)
            
            if patterns_embeddings is not None:
                await cl.Message(content="تم تجهيز النظام بنجاح! كيف يمكنني مساعدتك اليوم؟").send()
            else:
                await cl.Message(content="تعذر الاتصال بخدمة النماذج. يرجى التأكد من مفتاح HF_TOKEN.").send()


@cl.on_message
async def main(message: cl.Message):
    """الدالة الأساسية لمعالجة رسائل المستخدم"""
    user_text = message.content

    if patterns_embeddings is None:
        await cl.Message(content="النظام غير جاهز حالياً، يرجى إعادة تحديث الصفحة.").send()
        return

    try:
        # حساب متجه رسالة المستخدم
        user_embedding = get_embeddings_from_hf([user_text])

        if user_embedding is None:
            await cl.Message(content="حدث خطأ أثناء التواصل مع نموذج الذكاء الاصطناعي.").send()
            return

        user_vec = user_embedding[0]

        # حساب التشابه الدلالي (Cosine Similarity)
        similarities = np.dot(patterns_embeddings, user_vec)

        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        THRESHOLD = 0.50  # عتبة الثقة المناسبة لـ MiniLM

        if best_score >= THRESHOLD:
            matched_intent = intents_mapping[best_match_idx]
            selected_response = random.choice(matched_intent["responses"])
        else:
            selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"

    except Exception as e:
        print(f"خطأ أثناء المعالجة: {e}")
        selected_response = "حدث خطأ غير متوقع أثناء معالجة رسالتك."

    await cl.Message(content=selected_response).send()
