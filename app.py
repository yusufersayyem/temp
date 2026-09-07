import chainlit as cl
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@cl.on_chat_start
async def start():
    # تحميل ملف المعرفة الخفيف المصمم لـ TF-IDF
    with open('chatbot_knowledge.pkl', 'rb') as f:
        knowledge = cl.user_session.get("knowledge", None)
        if not knowledge:
            knowledge = pickle.load(f)
            cl.user_session.set("knowledge", knowledge)

    await cl.Message(
        content="مرحباً بك في المساعد الذكي لمديرية تربية نينوى! كيف يمكنني مساعدتك اليوم؟"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    knowledge = cl.user_session.get("knowledge")
    
    vectorizer = knowledge["vectorizer"]
    tfidf_matrix = knowledge["tfidf_matrix"]
    
    # تحويل نص سؤال المستخدم باستخدام الـ Vectorizer
    user_query = message.content
    query_vector = vectorizer.transform([user_query])
    
    # حساب درجة التشابه Cosine Similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    
    # اختيار أفضل مطابقة
    best_match_idx = np.argmax(similarities)
    best_score = similarities[best_match_idx]
    
    # ملاحظة: نسبة التشابه في TF-IDF تكون ممتازة عندما تتجاوز 0.25 إلى 0.30
    if best_score > 0.25:
        response = knowledge['responses'][best_match_idx]
    else:
        response = "عذراً، لم أفهم سؤالك بشكل دقيق. هل يمكنك إعادة صياغته؟ أو الاستفسار عن الموارد البشرية، الإجازات، أو المعاملات الرسمية في تربية نينوى."
        
    await cl.Message(content=response).send()
