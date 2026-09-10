import json
import random
import pickle
import chainlit as cl

# تحميل النموذج والـ Vectorizer وملف Intents عند بدء التشغيل
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

def get_response(user_input):
    # تحويل النص إلى متجهات رقمية
    input_vector = vectorizer.transform([user_input])
    
    # التنبؤ بالـ tag والاحتمالية
    probabilities = model.predict_proba(input_vector)[0]
    max_prob_index = probabilities.argmax()
    predicted_tag = model.classes_[max_prob_index]
    confidence = probabilities[max_prob_index]

    # إذا كانت نسبة الثقة أقل من 40% نعيد إجابة افتراضية
    if confidence < 0.4:
        return "عذراً، لم أفهم قصدك بوضوح. هل يمكنك إعادة الصياغة؟"

    # البحث عن الـ tag وإرجاع إجابة عشوائية من القائمة
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
            
    return "عذراً، حدث خطأ غير متوقع."

@cl.on_chat_start
async def start():
    await cl.Message(content="أهلاً بك! كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def main(message: cl.Message):
    response = get_response(message.content)
    await cl.Message(content=response).send()
