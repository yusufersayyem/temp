import json
import random
import re
import chainlit as cl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. قائمة الكلمات المستبعدة
RAW_ARABIC_STOP_WORDS = [
    "من", "في", "على", "إلى", "الي", "عن", "حتى", "مع", "هذا", "هذه", "هؤلاء", "ذلك", "تلك",
    "هل", "ما", "ماذا", "منذ", "كيف", "متى", "أين", "اين", "كم", "لماذا", "أي", "اي",
    "أن", "ان", "إن", "كان", "كانت", "يكون", "التي", "الذي", "الذين", "اللاتي", "قد",
    "اريد", "أريد", "ممكن", "سمحت", "بالله", "عفوا", "الله", "بس", "يعني"
]

# تنظيف الكلمات المستبعدة أولاً
def basic_normalize(text):
    text = re.sub(r"[\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"[إأآا]", "ا", text)          # توحيد الألف
    text = re.sub(r"ى", "ي", text)               # توحيد الياء
    text = re.sub(r"ة", "ه", text)               # توحيد التاء المربوطة
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

ARABIC_STOP_WORDS = set([basic_normalize(w) for w in RAW_ARABIC_STOP_WORDS if w])

# 2. دالة تنظيف النص وإزالة الكلمات الزائدة
def normalize_and_clean_arabic(text):
    text = basic_normalize(text)
    
    # حذف الكلمات المستبعدة يدوياً لضمان عملها مع char_wb
    words = text.split()
    filtered_words = [w for w in words if w not in ARABIC_STOP_WORDS]
    
    # إذا حُذفت كل الكلمات (مثلاً لو كتب المستخدم "من في") نرجع النص الأصلي المنظف
    result = " ".join(filtered_words)
    return result if result else text

patterns_list = []
intents_mapping = []
vectorizer = None
tfidf_matrix = None

def load_and_prepare_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, vectorizer, tfidf_matrix
    
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
            
    # إعداد المحرك: ngram_range من (2, 4) يعطي مرونة أكبر مع العامية والأخطاء الإملائية
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(patterns_list)
    print("تم إعداد محرك البحث النصي بنجاح!")

load_and_prepare_intents()

@cl.on_message
async def main(message: cl.Message):
    user_text = normalize_and_clean_arabic(message.content)
    
    if not user_text:
        await cl.Message(content="لطفاً، اكتب سؤالاً واضحاً.").send()
        return

    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]
    
    # تعديل العتبة إلى 0.30 لضمان استجابة أفضل للأسئلة القصيرة جداً
    THRESHOLD = 0.30
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة صياغة السؤال؟"
        
    await cl.Message(content=selected_response).send()
