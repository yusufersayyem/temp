import json
import random
import re
import chainlit as cl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. تنظيف النص العربي بطريقة متوازنة
def normalize_arabic(text):
    text = re.sub(r"[\u064B-\u0652]", "", text)  # إزالة التشكيل
    text = re.sub(r"[إأآا]", "ا", text)          # توحيد الألف
    text = re.sub(r"ى", "ي", text)               # توحيد الياء
    text = re.sub(r"ؤ", "ء", text)
    text = re.sub(r"ئ", "ء", text)
    text = re.sub(r"ة", "ه", text)               # توحيد التاء المربوطة
    text = re.sub(r"[^\w\s]", " ", text)         # استبدال الترقيم بمسافة
    return re.sub(r"\s+", " ", text).strip().lower()

# 2. قائمة الكلمات المستبعدة (كلمات منفردة فقط)
RAW_ARABIC_STOP_WORDS = [
    "من", "في", "على", "إلى", "الي", "عن", "حتى", "مع", "هذا", "هذه", "هؤلاء", "ذلك", "تلك",
    "هل", "ما", "ماذا", "منذ", "كيف", "متى", "أين", "اين", "كم", "لماذا", "أي", "اي",
    "أن", "ان", "إن", "كان", "كانت", "يكون", "التي", "الذي", "الذين", "اللاتي", "قد",
    "اريد", "أريد", "ممكن", "شلون", "شنو", "وين", "شوقت", "ليش", "هسه",
    "منو", "يابا", "سمحت", "بالله", "عفوا", "الله", "بس", "يعني"
]

ARABIC_STOP_WORDS = list(set([normalize_arabic(w) for w in RAW_ARABIC_STOP_WORDS if w]))

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
            clean_pattern = normalize_arabic(pattern)
            patterns_list.append(clean_pattern)
            intents_mapping.append({
                "tag": tag,
                "responses": responses
            })
            
    # استخدام char_wb (Character n-grams inside word boundaries)
    # يمنح دقة عالية جداً للغة العربية والعاميات ومقاومة الأخطاء الإملائية
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        stop_words=None  # stop_words تُهمل عند استخدام char_wb ولكن التنظيف المسبق يكفي
    )
    
    tfidf_matrix = vectorizer.fit_transform(patterns_list)
    print("تم إعداد محرك البحث النصي بنجاح!")

load_and_prepare_intents()

@cl.on_message
async def main(message: cl.Message):
    user_text = normalize_arabic(message.content)
    
    if not user_text:
        await cl.Message(content="لطفاً، اكتب سؤالاً واضحاً.").send()
        return

    user_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()
    
    best_match_idx = similarities.argmax()
    best_score = similarities[best_match_idx]
    
    # مع char_wb تكون العتبة المناسبة عادة بين 0.35 و 0.45
    THRESHOLD = 0.38
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة صياغة السؤال؟"
        
    await cl.Message(content=selected_response).send()
