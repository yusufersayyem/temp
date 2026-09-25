import json
import random
import re
import os
import chainlit as cl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_mistralai import MistralAIEmbeddings

# 1. قائمة الكلمات المستبعدة
RAW_ARABIC_STOP_WORDS = [
    "من", "في", "على", "إلى", "الي", "عن", "حتى", "مع", "هذا", "هذه", "هؤلاء", "ذلك", "تلك",
    "هل", "ما", "ماذا", "منذ", "كيف", "متى", "أين", "اين", "كم", "لماذا", "أي", "اي",
    "أن", "ان", "إن", "كان", "كانت", "يكون", "التي", "الذي", "الذين", "اللاتي", "قد",
    "اريد", "أريد", "ممكن", "سمحت", "بالله", "عفوا", "الله", "بس", "يعني"
]

def basic_normalize(text):
    text = re.sub(r"[\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"[إأآا]", "ا", text)          # توحيد الألف
    text = re.sub(r"ى", "ي", text)               # توحيد الياء
    text = re.sub(r"ة", "ه", text)               # توحيد التاء المربوطة
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

ARABIC_STOP_WORDS = set([basic_normalize(w) for w in RAW_ARABIC_STOP_WORDS if w])

def normalize_and_clean_arabic(text):
    text = basic_normalize(text)
    words = text.split()
    filtered_words = [w for w in words if w not in ARABIC_STOP_WORDS]
    result = " ".join(filtered_words)
    return result if result else text

# المتغيرات العامة
patterns_list = []
intents_mapping = []
embeddings_model = None
patterns_embeddings = None

def load_and_prepare_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, embeddings_model, patterns_embeddings
    
    # تأكد من ضبط مفتاح API الخاص بـ Mistral في بيئة التشغيل
    # os.environ["MISTRAL_API_KEY"] = "your-api-key-here"
    
    embeddings_model = MistralAIEmbeddings(model="mistral-embed")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            clean_pattern = normalize_and_clean_arabic(pattern)
            patterns_list.append(clean_pattern)
            intents_mapping.append({
                "tag": tag,
                "responses": responses
            })
            
    # توليد التضمينات لكل الأنماط المخزنة
    print("جاري إنشاء التضمينات بواسطة Mistral Embeddings...")
    patterns_embeddings = np.array(embeddings_model.embed_documents(patterns_list))
    print("تم إعداد محرك البحث بنجاح!")

load_and_prepare_intents()

@cl.on_chat_start
async def on_chat_start():
    pass

@cl.on_message
async def main(message: cl.Message):
    user_text = normalize_and_clean_arabic(message.content)
    
    if not user_text:
        await cl.Message(content="لطفاً، اكتب سؤالاً واضحاً.").send()
        return

    # استخراج تضمين السؤال الخاص بالمستخدم
    user_vector = np.array(embeddings_model.embed_query(user_text)).reshape(1, -1)
    
    # حساب نسبة التشابه
    similarities = cosine_similarity(user_vector, patterns_embeddings).flatten()
    
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]
    
    # تم تعديل العتبة لـ 0.60 لأن نماذج Embeddings تعطي قيم تشابه أعلى مقارنة بـ TF-IDF
    THRESHOLD = 0.60
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة صياغة السؤال؟"
        
    await cl.Message(content=selected_response).send()
