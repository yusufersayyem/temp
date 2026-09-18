import os
import json
import random
import cohere
import numpy as np
import chainlit as cl

# 1. تهيئة عميل Cohere باستخدام مفتاح البيئة
COHERE_API_KEY = os.environ.get("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# متغيرات التخزين العامة للمتجهات
patterns_list = []
intents_mapping = []
patterns_embeddings = None

# 2. قائمة الإعلانات (يمكنك تعديل النصوص والعناوين وروابط الصور)
ADS_LIST = [
    {
        "title": "📢 إعلان: خصم 20% على الدورات البرمجية!",
        "image_url": "https://via.placeholder.com/600x200?text=Ad+1",
        "link": "https://example.com/ad1"
    },
    {
        "title": "🚀 إعلان: استضف مشاريعك بسهولة مع خدماتنا السحابية!",
        "image_url": "https://via.placeholder.com/600x200?text=Ad+2",
        "link": "https://example.com/ad2"
    },
    {
        "title": "💡 إعلان: اشترك الآن في النشرة البرمجية اليومية!",
        "image_url": "https://via.placeholder.com/600x200?text=Ad+3",
        "link": "https://example.com/ad3"
    }
]

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
    
    # استخدام النموذج المتعدد اللغات المعتمد
    response = co.embed(
        texts=patterns_list,
        model="embed-multilingual-v3.0",
        input_type="search_document"
    )
    
    # تحويل النتيجة إلى NumPy Array ومعايرة المتجهات (Normalization)
    embeddings_matrix = np.array(response.embeddings)
    patterns_embeddings = embeddings_matrix / np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    print("تم تجهيز متجهات الـ patterns بنجاح!")

# تشغيل عملية حساب المتجهات مسبقاً عند بدء إقلاع التطبيق
load_and_embed_intents()

@cl.on_chat_start
async def start():
    # تهيئة العداد ومؤشر الإعلان لكل مستخدم بشكل مستقل
    cl.user_session.set("message_count", 0)
    cl.user_session.set("ad_index", 0)

@cl.on_message
async def main(message: cl.Message):
    # جلب وتحديث العدادات الخاصة بالمستخدم
    count = cl.user_session.get("message_count", 0) + 1
    ad_index = cl.user_session.get("ad_index", 0)
    cl.user_session.set("message_count", count)

    user_text = message.content
    
    # تحويل نص المستخدم إلى متجه عبر API
    user_response = co.embed(
        texts=[user_text],
        model="embed-multilingual-v3.0",
        input_type="search_query"
    )
    
    user_embedding = np.array(user_response.embeddings[0])
    user_embedding = user_embedding / np.linalg.norm(user_embedding)
    
    # حساب التشابه الدلالي (Cosine Similarity)
    similarities = np.dot(patterns_embeddings, user_embedding)
    
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    # عتبة الثقة (Threshold)
    THRESHOLD = 0.40
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"

    # التحقق من الشرط (كل 3 استعلامات)
    elements = []
    if count % 3 == 0:
        current_ad = ADS_LIST[ad_index]
        
        # إرفاق صورة الإعلان
        elements.append(
            cl.Image(
                name=f"ad_{ad_index + 1}",
                url=current_ad["image_url"],
                display="inline"
            )
        )
        
        # إضافة نص الإعلان والرابط أسفل رد البوت
        selected_response += f"\n\n---\n{current_ad['title']}\n[اضغط هنا للمزيد]({current_ad['link']})"
        
        # الانتقال للإعلان التالي بالتتابع (0 -> 1 -> 2 -> 0)
        next_ad_index = (ad_index + 1) % len(ADS_LIST)
        cl.user_session.set("ad_index", next_ad_index)

    # إرسال الرسالة النهائية
    await cl.Message(content=selected_response, elements=elements).send()
