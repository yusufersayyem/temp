import json
import random
import numpy as np
import chainlit as cl
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. تحميل نموذج التضمين متعدد اللغات (يدعم العربية بجودة عالية)
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 2. تحميل ملف Intents
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# 3. تجهيز بيانات التدريب وتحويل الأنماط إلى Embeddings
patterns = []       # قائمة تحتوي على جميع الأسئلة/الأنماط
pattern_tags = []   # التصنيف المقابل لكل سؤال

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        pattern_tags.append(intent['tag'])

# تحويل كافة الأنماط إلى Embeddings وتخزينها في الذاكرة
patterns_embeddings = embedder.encode(patterns, convert_to_numpy=True)


def get_response_by_tag(tag):
    """جلب رد عشوائي للنية المحددة"""
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "عذراً، لم أفهم ما تقصده."


@cl.on_chat_start
async def start():
    await cl.Message(content="أهلاً بك! أنا أعمل الآن بالذكاء الاصطناعي والدلالة اللفظية. كيف يمكنني مساعدتك؟").send()


@cl.on_message
async def main(message: cl.Message):
    user_text = message.content.strip()

    # تحويل نص المستخدم إلى Embedding
    user_embedding = embedder.encode([user_text], convert_to_numpy=True)

    # حساب جيب تمام التشابه (Cosine Similarity) بين نص المستخدم والأنماط المخزنة
    similarities = cosine_similarity(user_embedding, patterns_embeddings)[0]

    # معرفة أعلى درجة تشابه والمؤشر الخاص بها
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    predicted_tag = pattern_tags[best_match_idx]

    # تحديد حد أدنى للقبول (Threshold)
    SIMILARITY_THRESHOLD = 0.60

    if best_score < SIMILARITY_THRESHOLD:
        response = "عذراً، لم أفهم ما تقصده بوضوح. هل يمكنك إعادة الصياغة؟"
    else:
        response = get_response_by_tag(predicted_tag)

    await cl.Message(content=response).send()
