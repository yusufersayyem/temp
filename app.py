import os
import json
import random
import requests
import numpy as np
import chainlit as cl
from sklearn.metrics.pairwise import cosine_similarity

# 1. إعداد رابط النموذج ومفتاح API
# نستخدم نموذج خفيف وسريع يدعم اللغة العربية
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# جلب المفتاح من متغيرات البيئة (Environment Variables)
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

def query_embedding(texts):
    """دالة لإرسال النصوص لـ Hugging Face واستلام الـ Embeddings"""
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return np.array(response.json())
    else:
        raise Exception(f"خطأ في API: {response.status_code} - {response.text}")

# 2. تحميل البيانات وتجهيز الأسئلة
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

patterns = []
pattern_tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        pattern_tags.append(intent['tag'])

# 3. حساب التضمينات للأنماط المجهزة عند بدء التشغيل
try:
    patterns_embeddings = query_embedding(patterns)
except Exception as e:
    print(f"حدث خطأ أثناء تحميل التضمينات الأولية: {e}")

def get_response_by_tag(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "عذراً، لم أفهم ما تقصده."

@cl.on_chat_start
async def start():
    await cl.Message(content="أهلاً بك! كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def main(message: cl.Message):
    user_text = message.content.strip()

    try:
        # تحويل نص المستخدم إلى Embedding عبر الـ API
        user_embedding = query_embedding([user_text])

        # حساب جيب تمام التشابه (Cosine Similarity)
        similarities = cosine_similarity(user_embedding, patterns_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score < 0.60:
            response = "عذراً، لم أفهم ما تقصده بوضوح. هل يمكنك إعادة الصياغة؟"
        else:
            response = get_response_by_tag(pattern_tags[best_match_idx])

    except Exception as e:
        response = "حدث خطأ أثناء التواصل مع خادم الذكاء الاصطناعي، يرجى المحاولة لاحقاً."

    await cl.Message(content=response).send()
