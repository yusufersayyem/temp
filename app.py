import json
import random
import re
import os
import chainlit as cl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_mistralai import MistralAIEmbeddings

# 1. قائمة الكلمات المستبعدة (Stop Words)
RAW_ARABIC_STOP_WORDS = [
    "من", "في", "على", "إلى", "الي", "عن", "حتى", "مع", "هذا", "هذه", "هؤلاء", "ذلك", "تلك",
    "هل", "ما", "ماذا", "منذ", "كيف", "متى", "أين", "اين", "كم", "لماذا", "أي", "اي",
    "أن", "ان", "إن", "كان", "كانت", "يكون", "التي", "الذي", "الذين", "اللاتي", "قد",
    "اريد", "أريد", "ممكن", "سمحت", "بالله", "عفوا", "الله", "بس", "يعني"
]

def basic_normalize(text):
    if not text:
        return ""
    text = re.sub(r"[\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"[إأآا]", "ا", text)          # توحيد الألف
    text = re.sub(r"ى", "ي", text)               # توحيد الياء
    text = re.sub(r"ة", "ه", text)               # توحيد التاء المربوطة
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

ARABIC_STOP_WORDS = set([basic_normalize(w) for w in RAW_ARABIC_STOP_WORDS if w])

# 2. دالة تنظيف النص مع حماية ضد النصوص الفارغة
def normalize_and_clean_arabic(text):
    normalized = basic_normalize(text)
    words = normalized.split()
    filtered_words = [w for w in words if w not in ARABIC_STOP_WORDS]
    result = " ".join(filtered_words)
    
    # إذا كانت النتيجة فارغة بعد إزالة الكلمات المستبعدة، نستخدم النص المعالج أوليًا لمنع الإرسال الفارغ لـ Mistral
    return result if result else normalized

# المتغيرات العامة
patterns_list = []
intents_mapping = []
embeddings_model = None
patterns_embeddings = None

def load_and_prepare_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, embeddings_model, patterns_embeddings
    
    # تهيئة نموذج التضمين الخاص بـ Mistral
    embeddings_model = MistralAIEmbeddings(model="mistral-embed")
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            clean_pattern = normalize_and_clean_arabic(pattern)
            
            # حماية إضافية: التأكد من عدم إضافة أي نص فارغ إطلاقًا
            if not clean_pattern.strip():
                clean_pattern = pattern.strip()
                
            if clean_pattern:
                patterns_list.append(clean_pattern)
                intents_mapping.append({
                    "tag": tag,
                    "responses": responses
                })
            
    print("جاري إنشاء التضمينات بواسطة Mistral Embeddings...")
    # توليد متجهات التضمين للأنماط
    patterns_embeddings = np.array(embeddings_model.embed_documents(patterns_list))
    print("تم إعداد محرك البحث والنوايا بنجاح!")

# تحميل البيانات والنموذج عند بدء التشغيل
load_and_prepare_intents()

@cl.on_chat_start
async def on_chat_start():
    # تم إزالة كافة إعدادات الإعلانات من جلسة المستخدم
    pass

@cl.on_message
async def main(message: cl.Message):
    raw_input = message.content.strip()
    if not raw_input:
        await cl.Message(content="لطفاً، اكتب سؤالاً واضحاً.").send()
        return

    user_text = normalize_and_clean_arabic(raw_input)
    
    # إذا كانت النتيجة فارغة، نستخدم النص الأصلي لضمان عدم إرسال نص فارغ لـ Mistral API
    if not user_text.strip():
        user_text = raw_input

    # توليد التضمين الخاص بسؤال المستخدم
    user_vector = np.array(embeddings_model.embed_query(user_text)).reshape(1, -1)
    
    # حساب نسبة التشابه التشابهي (Cosine Similarity)
    similarities = cosine_similarity(user_vector, patterns_embeddings).flatten()
    
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]
    
    # تم رفع العتبة إلى 0.60 لأن التضمين الدلالي يعطي درجات تشابه أعلى مقارنة بـ TF-IDF
    THRESHOLD = 0.60
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة صياغة السؤال؟"
        
    await cl.Message(content=selected_response).send()
