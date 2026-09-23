import json
import os
import random
import time
import requests
import chainlit as cl
import numpy as np

# 1. إعداد مفتاح API والرابط الجديد لمعالج Hugging Face (Router API)
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "intfloat/multilingual-e5-small"

# استخدام الرابط المحدث والمدعوم حالياً من Hugging Face
API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_NAME}"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# متغيرات التخزين
patterns_list = []
intents_mapping = []
patterns_embeddings = None


def get_embeddings_from_hf(texts, retries=3, delay=5):
    """استدعاء Hugging Face API مع دعم إعادة المحاولة تلقائياً عند فشل الاتصال"""
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
            
            if response.status_code == 200:
                embeddings = np.array(response.json())
                
                # التأكد من صحة أبعاد المصفوفة
                if len(embeddings.shape) == 1:
                    embeddings = np.expand_dims(embeddings, axis=0)

                # معايرة المتجهات (Normalization)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                return embeddings / norms
            else:
                print(f"محاولة {attempt + 1}: فشل الاستجابة ({response.status_code}) - {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"محاولة {attempt + 1}: خطأ في الاتصال بالشبكة ({e})")
            
        if attempt < retries - 1:
            time.sleep(delay)

    raise RuntimeError("تعذر الاتصال بـ Hugging Face API بعد عدة محاولات.")


def load_and_embed_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, patterns_embeddings

    if not os.path.exists(json_path):
        print(f"خطأ: الملف {json_path} غير موجود!")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_patterns = []
    for intent in data.get("intents", []):
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            # إضافة البادئة passage: الخاصة بنموذج E5
            raw_patterns.append(f"passage: {pattern}")
            intents_mapping.append({"tag": tag, "responses": responses})

    print("جاري استدعاء Hugging Face API لحساب متجهات الـ patterns...")
    patterns_embeddings = get_embeddings_from_hf(raw_patterns)
    print("تم تجهيز متجهات الـ patterns بنجاح!")


# تحميل وحساب المتجهات عند بداية تشغيل التطبيق
try:
    load_and_embed_intents()
except Exception as e:
    print(f"تحذير: فشل حساب المتجهات عند التشغيل: {e}")


@cl.on_message
async def main(message: cl.Message):
    user_text = message.content

    if patterns_embeddings is None:
        await cl.Message(content="تطبيقي حالياً يواجه مشكلة في الاتصال بالنموذج، يرجى المحاولة بعد قليل.").send()
        return

    # إضافة البادئة query: لنص المستخدم
    formatted_user_text = f"query: {user_text}"

    try:
        # حساب متجه نص المستخدم
        user_embedding = get_embeddings_from_hf([formatted_user_text])[0]

        # حساب التشابه الدلالي (Cosine Similarity)
        similarities = np.dot(patterns_embeddings, user_embedding)

        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        THRESHOLD = 0.80

        if best_score >= THRESHOLD:
            matched_intent = intents_mapping[best_match_idx]
            selected_response = random.choice(matched_intent["responses"])
        else:
            selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"

    except Exception as e:
        print(f"خطأ أثناء معالجة الرسالة: {e}")
        selected_response = "حدث خطأ أثناء معالجة طلبك، يرجى إعادة المحاولة."

    await cl.Message(content=selected_response).send()
