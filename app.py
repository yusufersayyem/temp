import chainlit as cl
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# تحميل الملف مرة واحدة عند إقلاع السيرفر لتوفير الذاكرة والوقت
try:
    with open('chatbot_knowledge.pkl', 'rb') as f:
        GLOBAL_KNOWLEDGE = pickle.load(f)
    print("تم تحميل ملف المعرفة بنجاح!")
except Exception as e:
    print(f"خطأ في تحميل ملف PKL: {e}")
    GLOBAL_KNOWLEDGE = None

@cl.on_chat_start
async def start():
    if GLOBAL_KNOWLEDGE:
        cl.user_session.set("knowledge", GLOBAL_KNOWLEDGE)
    
    await cl.Message(
        content="مرحباً بك في المساعد الذكي لمديرية تربية نينوى! كيف يمكنني مساعدتك اليوم؟"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    knowledge = cl.user_session.get("knowledge")
    
    if not knowledge:
        await cl.Message(content="حدث خطأ في تحميل قاعدة المعرفة، يرجى المحاولة لاحقاً.").send()
        return

    vectorizer = knowledge["vectorizer"]
    tfidf_matrix = knowledge["tfidf_matrix"]
    
    # استخراج نص السؤال
    user_query = message.content.strip()
    
    try:
        # تحويل نص سؤال المستخدم
        query_vector = vectorizer.transform([user_query])
        
        # حساب درجة التشابه Cosine Similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
        
        # اختيار أفضل مطابقة
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        
        # تخفيض العتبة قليلاً لضمان قبول الإجابات المتطابقة جزئياً (مثلاً 0.15 أو 0.20)
        if best_score > 0.18:
            response = knowledge['responses'][best_match_idx]
        else:
            response = "عذراً، لم أفهم سؤالك بشكل دقيق. هل يمكنك إعادة صياغته؟ أو الاستفسار عن الموارد البشرية، الإجازات، أو المعاملات الرسمية في تربية نينوى."
            
    except Exception as e:
        response = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"

    await cl.Message(content=response).send()
