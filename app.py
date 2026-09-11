import os
import json
import random
import time
import requests
import numpy as np
import chainlit as cl
from sklearn.metrics.pairwise import cosine_similarity

# 1. إعداد رابط النموذج ومفتاح الـ Token
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

def query_embedding(texts, retries=3, delay=5):
    """دالة إرسال الطلبات لـ Hugging Face مع معالجة الأخطاء وإعادة المحاولة"""
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN غير موجود في متغيرات البيئة (Environment Variables).")
        
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": texts, "options": {"wait_for_model": True}}
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                res_np = np.array(result)
                if res_np.ndim == 3:
                    res_np = res_np.mean(axis=1)
                return res_np
            elif response.status_code == 503:
                print(f"النموذج قيد التحميل... محاولة {attempt + 1} من {retries}")
                time.sleep(delay)
            else:
                print(f"HF API Error: {response.status_code} - {response.text}")
                break
        except Exception as e:
            print(f"Request Exception: {e}")
            time.sleep(delay)
            
    raise Exception("فشل الاتصال بـ Hugging Face API.")

# 2. قراءة ملف البيانات
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

patterns = []
pattern_tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        pattern_tags.append(intent['tag'])

# المتغير الذي سيحفظ التضمينات المخزنة
patterns_embeddings = None

def load_embeddings():
    """حساب أو جلب التضمينات عند الحاجة دون إيقاف السيرفر عند الإقلاع"""
    global patterns_embeddings
    if patterns_embeddings is None:
        patterns_embeddings = query_embedding(patterns)
    return patterns_embeddings

def get_response_by_tag(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "عذراً، لم أفهم ما تقصده."

# 3. أحداث Chainlit (يجب أن يتم الوصول إليها حتى يتأكد Chainlit من وجودها)
@cl.on_chat_start
async def start():
    # محاولة تجهيز التضمينات في خلفية أول محادثة
    try:
        load_embeddings()
    except Exception as e:
        print(f"Warning on startup embedding load: {e}")
    await cl.Message(content="أهلاً بك! كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def main(message: cl.Message):
    user_text = message.content.strip()

    try:
        # التأكد من جاهزية التضمينات المخزنة
        p_embeddings = load_embeddings()
        
        # تحويل نص المستخدم الحالي
        user_embedding = query_embedding([user_text])

        # حساب التشابه
        similarities = cosine_similarity(user_embedding, p_embeddings)[0]
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]

        if best_score < 0.60:
            response = "عذراً، لم أفهم ما تقصده بوضوح. هل يمكنك إعادة الصياغة؟"
        else:
            response = get_response_by_tag(pattern_tags[best_match_idx])

    except Exception as e:
        print(f"Error in main loop: {e}")
        response = "حدث خطأ أثناء التواصل مع الذكاء الاصطناعي، يرجى التأكد من إضافة HF_TOKEN في Render وتجربة المحاولة لاحقاً."

    await cl.Message(content=response).send()
