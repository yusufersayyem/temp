import json
import random
import re
import chainlit as cl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# قائمة الإعلانات (يمكن استخدام مسار محلي local path أو رابط URL مباشر للصورة)
ADS = [
    {
        "title": "معهد لارسا النموذجي - خصم 20% على جميع الدورات!",
        "image_url": "https://ik.imagekit.io/63rncvror/ad6.webp?updatedAt=1785601370285?random=1",  # استبدل هذا برابط صورتك أو مسارها المحلي مثل "ads/ad1.jpg"
        "target_url": "https://www.facebook.com/larsafoundation/"
    },
    {
        "title": "خصم خاص 20% لموظفي التربية",
        "image_url": "https://ik.imagekit.io/63rncvror/ad1.webp?updatedAt=1785601369756?random=2",
        "target_url": "https://www.facebook.com/khutarrest/?locale=ku_TR"
    },
    {
        "title": "حمل تطبيقنا الجديد للوصول إلى كافة الخدمات",
        "image_url": "https://ik.imagekit.io/63rncvror/ad5.webp?updatedAt=1785601364212?random=3",
        "target_url": "https://www.asiacell.com/personal?gad_source=1&gad_campaignid=21900349889&gbraid=0AAAAAoo1Wz11yBTnlEw-9ZZxIQFMgUlc_&gclid=Cj0KCQjwt9jVBhDXARIsAFSP-6ercxBv9pDtXeaI8gIK-aKaBCsRCBRtigCYoQBavnMAWeKr9xoEETEaAi9iEALw_wcB"
    },
    {
        "title": "شارك البوت مع أصدقائك واحصل على خصم خاص",
        "image_url": "https://picsum.photos/1200/600?random=4",
        "target_url": "https://example.com/share"
    },
    {
        "title": "تواصل معنا مباشرة عبر الواتساب للاعلان على البرنامج",
        "image_url": "https://ik.imagekit.io/63rncvror/ads.jpg?random=5"
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
    # زيادة عداد الاستعلامات للمستخدم
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
        
    # عرض الإعلان عند كل ثالث استعلام
    if query_count % 3 == 0:
        ad_index = cl.user_session.get("ad_index", 0)
        ad = ADS[ad_index]
        
        # إنشاء عنصر الصورة المباشر من Chainlit
        image_element = cl.Image(
            url=ad["image_url"],  # أو استخدم path="path/to/image.jpg" للصور المحلية
            name=ad["title"],
            display="inline",
            size="large"
        )
        
        # نص الإعلان مع رابط التوجيه عند الضغط
        ad_text = f"\n\n---\n📢 **إعلان**\n[{ad['title']}]({ad['target_url']})"
        full_response = f"{selected_response}{ad_text}"
        
        # إرسال الرسالة مع الصورة بدون الفراغات الجانبية
        await cl.Message(content=full_response, elements=[image_element]).send()
        
        # التدوير للإعلان التالي
        next_ad_index = (ad_index + 1) % len(ADS)
        cl.user_session.set("ad_index", next_ad_index)
    else:
        await cl.Message(content=selected_response).send()
