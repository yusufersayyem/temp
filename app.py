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
        # جلب رابط أو مسار الصورة إن وجد، وإلا نجعله None
        image_path = intent.get("image", None)
        
        for pattern in intent["patterns"]:
            clean_pattern = normalize_and_clean_arabic(pattern)
            patterns_list.append(clean_pattern)
            intents_mapping.append({
                "tag": tag,
                "responses": responses,
                "image": image_path
            })
            
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(patterns_list)
    print("تم إعداد محرك البحث النصي مع دعم الصور بنجاح!")

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
    
    THRESHOLD = 0.30
    
    if best_score >= THRESHOLD:
        matched_intent = intents_mapping[best_match_idx]
        selected_response = random.choice(matched_intent["responses"])
        image_url = matched_intent.get("image")
        
        elements = []
        if image_url:
            # التحقق مما إذا كان الرابط يبدأ بـ http لتحديد ما إذا كان رابطاً خارجيئاً أو مساراً محلياً
            if image_url.startswith("http://") or image_url.startswith("https://"):
                elements.append(
                    cl.Image(name="intent_image", url=image_url, display="inline")
                )
            else:
                elements.append(
                    cl.Image(name="intent_image", path=image_url, display="inline")
                )
            
        await cl.Message(
            content=selected_response,
            elements=elements
        ).send()
        
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة صياغة السؤال؟"
        await cl.Message(content=selected_response).send()
