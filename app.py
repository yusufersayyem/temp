import json
import pickle
import random
import chainlit as cl

# تحميل النموذج وحمولة الـ JSON عند بدء التشغيل
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# دالة للحصول على الرد بناءً على الـ tag المتوقع
def get_response(predicted_tag):
    for intent in intents['intents']:
        if intent['tag'] == predicted_tag:
            return random.choice(intent['responses'])
    return "عذراً، لم أفهم ما تقصده."

@cl.on_chat_start
async def start():
    await cl.Message(content="أهلاً بك! كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def main(message: cl.Message):
    user_text = message.content.strip()
    
    # التنبؤ بالـ tag
    predicted_tag = model.predict([user_text])[0]
    
    # جلب الرد
    response = get_response(predicted_tag)
    
    await cl.Message(content=response).send()