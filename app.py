import json
import os
import random
import requests
import chainlit as cl
import numpy as np

# 1. إعداد مفتاح API واسم النموذج
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "intfloat/multilingual-e5-small"
API_URL = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_NAME}"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# متغيرات التخزين
patterns_list = []
intents_mapping = []
patterns_embeddings = None


def get_embeddings_from_hf(texts):
    """دالة مساعدة لاستدعاء Hugging Face Inference API مباشرة عبر requests"""
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": texts, "options": {"wait_for_model": True}}
    )
    
    # التأكد من نجاح الطلب
    if response.status_code != 200:
        raise RuntimeError(f"Hugging Face API Error ({response.status_code}): {response.text}")

    embeddings = np.array(response.json())

    # إذا تم إرسال نص واحد، نتأكد من إرجاع مصفوفة ثنائية الأبعاد
    if len(embeddings.shape) == 1:
        embeddings = np.expand_dims(embeddings, axis=0)

    # معايرة المتجهات (Normalization) لحساب Cosine Similarity مباشرة عبر Dot Product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # تفادي القسمة على صفر في حال وجود مصفوفات فارغة
    norms[norms == 0] = 1e-10
    return embeddings / norms


def load_and_embed_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, patterns_embeddings

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_patterns = []
    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            # إضافة البادئة passage: الخاصة بنموذج E5
            raw_patterns.append(f"passage: {pattern}")
            intents_mapping.append({"tag": tag, "responses": responses})

    print("جاري استدعاء Hugging Face API لحساب متجهات الـ patterns...")

    # طلب حساب المتجهات عبر Hugging Face Serverless API
    patterns_embeddings = get_embeddings_from_hf(raw_patterns)

    print("تم تجهيز متجهات الـ patterns بنجاح!")


# تشغيل عملية حساب المتجهات عند بداية التشغيل
load_and_embed_intents()


@cl.on_message
async def main(message: cl.Message):
    user_text = message.content

    # إضافة البادئة query: لنص المستخدم
    formatted_user_text = f"query: {user_text}"

    # حساب متجه نص المستخدم عبر Hugging Face API
    user_embedding = get_embeddings_from_hf([formatted_user_text])[0]

    # حساب التشابه الدلالي (Cosine Similarity)
    similarities = np.dot(patterns_embeddings, user_embedding)

    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    # عتبة الثقة (تعديل القيمة بما يتناسب مع نتائج E5)
    THRESHOLD = 0.80

    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = (
            "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"
        )

    await cl.Message(content=selected_response).send()
