import json
import random
import re
import chainlit as cl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. دالة لتنظيف وتطبيع النص العربي (مهمة جداً للغة العربية)
def normalize_arabic(text):
    text = re.sub(r"[\u064B-\u0652]", "", text) # إزالة التشكيل
    text = re.sub(r"[إأآا]", "ا", text)          # توحيد الألف
    text = re.sub(r"ى", "ي", text)               # توحيد الياء
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)               # توحيد التاء المربوطة والهاء
    text = re.sub(r"[^\w\s]", "", text)          # إزالة علامات الترقيم
    return text.lower().strip()

patterns_list = []
intents_mapping = []
vectorizer = None
tfidf_matrix = None

# 2. تحميل البيانات وتدريب محرك البحث المحلي (TF-IDF)
def load_and_prepare_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, vectorizer, tfidf_matrix
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            # تنظيف النماذج قبل تخزينها
            clean_pattern = normalize_arabic(pattern)
            patterns_list.append(clean_pattern)
            intents_mapping.append({
                "tag": tag,
                "responses": responses
            })
            
    # بناء نموذج TF-IDF على مستوى الحروف والكلمات (Char/Word n-grams) ليدعم العامية والأخطاء الإملائية
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf_matrix = vectorizer.fit_transform(patterns_list)
    print("تم إعداد محرك البحث النصي المحلي بنجاح!")

# تشغيل الإعداد عند بدء التطبيق
load_and_prepare_intents()

@cl.on_message
async def main(message: cl.Message):
    user_text = normalize_arabic(message.content)
    
    # تحويل نص المستخدم إلى متجه نصي محلياً
    user_vector = vectorizer.transform([user_text])
    
    # حساب التشابه بين استعلام المستخدم وكل الـ patterns
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]
    
    # عتبة الثقة (يمكنك تعديلها حسب تجربتك، عادة 0.25 إلى 0.35 ممتازة لـ TF-IDF)
    THRESHOLD = 0.28
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"
        
    await cl.Message(content=selected_response).send()
