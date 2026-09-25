import json
import random
import re
import chainlit as cl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# قائمة الإعلانات (روابط صور مع عنوان ورابط عند الضغط)
ADS = [
    {
        "title": "خصم 20% على جميع الدورات!",
        "image_url": "https://ik.imagekit.io/63rncvror/ad6.webp?updatedAt=1785601370285?random=1",
        "target_url": "https://example.com/offer1"
    },
    {
        "title": "اشترك في نشرتنا البريدية لتصلك أحدث الأخبار",
        "image_url": "https://ik.imagekit.io/63rncvror/ad7.webp?updatedAt=1785601364077?random=2",
        "target_url": "https://example.com/newsletter"
    },
    {
        "title": "حمل تطبيقنا الجديد الآن",
        "image_url": "https://ik.imagekit.io/63rncvror/ad10.webp?updatedAt=1785601362911?random=3",
        "target_url": "https://example.com/app"
    },
    {
        "title": "شارك البوت مع أصدقائك واحصل على مكافآت",
        "image_url": "https://ik.imagekit.io/63rncvror/ad3.webp?updatedAt=1785601369079?random=4",
        "target_url": "https://example.com/share"
    },
    {
        "title": "تقييمك يهمنا لتطوير الخدمة",
        "image_url": "https://https://ik.imagekit.io/63rncvror/ad5.webp?updatedAt=1785601364212?random=5",
        "target_url": "https://example.com/feedback"
    }
]

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

# 2. دالة تنظيف النص وإزالة الكلمات الزائدة
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
        for pattern in intent["patterns"]:
            clean_pattern = normalize_and_clean_arabic(pattern)
            patterns_list.append(clean_pattern)
            intents_mapping.append({
                "tag": tag,
                "responses": responses
            })
            
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        sublinear_tf=True
    )
    
    tfidf_matrix = vectorizer.fit_transform(patterns_list)
    print("تم إعداد محرك البحث النصي بنجاح!")

load_and_prepare_intents()

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("query_count", 0)
    cl.user_session.set("ad_index", 0)

@cl.on_message
async def main(message: cl.Message):
    # زيادة العداد لكل استعلام من المستخدم
    query_count = cl.user_session.get("query_count", 0) + 1
    cl.user_session.set("query_count", query_count)

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
    else:
        selected_response = "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة صياغة السؤال؟"
        
    # إظهار الإعلان عند كل ثالث استعلام
    if query_count % 3 == 0:
        ad_index = cl.user_session.get("ad_index", 0)
        ad = ADS[ad_index]
        
        # تنسيق الصورة ورابط الضغط بأسلوب Markdown:
        # [![نص بديل](رابط الصورة)](رابط التوجيه)
        ad_markdown = f"\n\n---\n📢 **إعلان**\n[{ad['title']}]({ad['target_url']})\n\n[![{ad['title']}]({ad['image_url']})]({ad['target_url']})"
        
        full_response = f"{selected_response}{ad_markdown}"
        
        # التدوير للإعلان التالي
        next_ad_index = (ad_index + 1) % len(ADS)
        cl.user_session.set("ad_index", next_ad_index)
    else:
        full_response = selected_response

    await cl.Message(content=full_response).send()
