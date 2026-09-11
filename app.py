import os
import json
import random
import cohere
import numpy as np
import chainlit as cl

# 1. تهيئة عميل Cohere باستخدام مفتاح البيئة
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# متغيرات التخزين
patterns_list = []
intents_mapping = []
patterns_embeddings = None

def load_and_embed_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, patterns_embeddings
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            patterns_list.append(pattern)
            intents_mapping.append({
                "tag": tag,
                "responses": responses
            })
            
    print("جاري استدعاء Cohere API لحساب متجهات الـ patterns...")
    
    # طلب الـ Embeddings عبر API بدعم ممتاز للعربية
    response = co.embed(
        texts=patterns_list,
        model="embed-arabic-v3.0",
        input_type="search_document"
    )
    
    # تحويل النتيجة إلى NumPy Array مع معايرة المتجهات
    embeddings_matrix = np.array(response.embeddings)
    # Norm / Normalization لحساب Cosine Similarity مباشرة عبر ضرب المصفوفات
    patterns_embeddings = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    print("تم التجهيز والربط بنجاح!")

# تشغيل التحضير عند إقلاع التطبيق
load_and_embed_intents()

@cl.on_message
async def main(message: cl.Message):
    user_text = message.content
    
    # تحويل رسالة المستخدم عبر API
    user_response = co.embed(
        texts=[user_text],
        model="embed-arabic-v3.0",
        input_type="search_query"
    )
    
    user_embedding = np.array(user_response.embeddings[0])
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    
    # حساب التشابه الدلالي (Cosine Similarity)
    similarities = np.dot(patterns_embeddings, user_embedding)
    
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    # عتبة الثقة (نموذج Cohere يعطي نواتج دقيقة، 0.40 عتبة ممتازة لهذا النموذج)
    THRESHOLD = 0.40
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"
        
    await cl.Message(content=selected_response).send()
