import json
import random
import numpy as np
import chainlit as cl
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# استخدام نموذج مصغر وفعال لتوفير الـ RAM على Render
embedder = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# تحميل الملفات وتحضير التضمينات
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

patterns = []
pattern_tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        pattern_tags.append(intent['tag'])

# حساب التضمينات مرة واحدة عند بدء التشغيل
patterns_embeddings = embedder.encode(patterns, convert_to_numpy=True)

def get_response_by_tag(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "عذراً، لم أفهم ما تقصده."

@cl.on_chat_start
async def start():
    await cl.Message(content="أهلاً بك! كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def main(message: cl.Message):
    user_text = message.content.strip()
    user_embedding = embedder.encode([user_text], convert_to_numpy=True)

    similarities = cosine_similarity(user_embedding, patterns_embeddings)[0]
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]

    if best_score < 0.55:
        response = "عذراً، لم أفهم ما تقصده بوضوح. هل يمكنك إعادة الصياغة؟"
    else:
        response = get_response_by_tag(pattern_tags[best_match_idx])

    await cl.Message(content=response).send()
