import json
import random
import re
import chainlit as cl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. دالة تنظيف وتطبيع النص العربي
def normalize_arabic(text):
    text = re.sub(r"[\u064B-\u0652]", "", text) # إزالة التشكيل
    text = re.sub(r"[إأآا]", "ا", text)          # توحيد الألف
    text = re.sub(r"ى", "ي", text)               # توحيد الياء
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)               # توحيد التاء المربوطة
    text = re.sub(r"[^\w\s]", "", text)          # إزالة علامات الترقيم
    return text.lower().strip()

# 2. قائمة الكلمات المستبعدة (Stop Words) باللغة العربية والعامية العراقية
RAW_ARABIC_STOP_WORDS = [
    # حروف وأدوات فصحى
    "من", "في", "على", "إلى", "الي", "عن", "حتى", "مع", "هذا", "هذه", "هؤلاء", "ذلك", "تلك",
    "هل", "ما", "ماذا", "منذ", "كيف", "متى", "أين", "اين", "كم", "لماذا", "أي", "اي",
    "أن", "ان", "إن", "كان", "كانت", "يكون", "التي", "الذي", "الذين", "اللاتي", "عن", "قد",
    # كلمات وعاميات شائعة في الأسئلة
    "اريد", "أريد", "اريد اعرف", "ممكن", "شلون", "شنو", "وين", "شوقت", "ليش", "هسه",
    "منو", "يابا", "لو سمحت", "بالله", "عفوا", "الله يخليك", "بس", "يعني"
]

# تنظيف قائمة الـ Stop Words لتطابق شكل النصوص المُدخلة
ARABIC_STOP_WORDS = [normalize_arabic(word) for word in RAW_ARABIC_STOP_WORDS]

patterns_list = []
intents_mapping = []
vectorizer = None
tfidf_matrix = None

# 3. تحميل البيانات وتدريب TF-IDF مع الـ Stop Words
def load_and_prepare_intents(json_path="intents.json"):
    global patterns_list, intents_mapping, vectorizer, tfidf_matrix
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    for intent in data["intents"]:
        tag = intent["tag"]
        responses = intent["responses"]
        for pattern in intent["patterns"]:
            clean_pattern = normalize_arabic(pattern)
            patterns_list.append(clean_pattern)
            intents_mapping.append({
                "tag": tag,
                "responses": responses
            })
            
    # تمرير stop_words واستخدام analyzer="word" للاستفادة الكاملة من تصفية الكلمات
    vectorizer = TfidfVectorizer(
        analyzer="word",
        stop_words=ARABIC_STOP_WORDS,
        ngram_range=(1, 2)  # يشمل الكلمات المنفردة والأزواج المفتاحية
    )
    
    tfidf_matrix = vectorizer.fit_transform(patterns_list)
    print("تم إعداد محرك البحث النصي المحلي واستبعاد الكلمات الشائعة بنجاح!")

load_and_prepare_intents()

@cl.on_message
async def main(message: cl.Message):
    user_text = normalize_arabic(message.content)
    
    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]
    
    # بعد استبعاد الـ Stop Words تكون النتيجة أكثر دقة ويمكن اعتماد عتبة بين 0.25 و 0.35
    THRESHOLD = 0.30
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"
        
    await cl.Message(content=selected_response).send()
