import json
import random
import re
import chainlit as cl
import joblib

# 1. دالة معالجة وتطبيع النص العربي
def normalize_arabic(text):
    text = re.sub(r'[\u064B-\u0652]', '', text)  # إزالة التشكيل
    text = re.sub(r'[إأآا]', 'ا', text)           # توحيد الألف
    text = re.sub(r'ى', 'ي', text)              # توحيد الياء
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)              # توحيد التاء المربوطة
    return text.strip()

# 2. تحميل النموذج والملفات (استخدام joblib بدلاً من pickle)
try:
    model = joblib.load('model.joblib')
except Exception:
    import pickle
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

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
    # معالجة النص المدخل
    cleaned_text = normalize_arabic(message.content)
    
    # حساب الاحتمالات لجميع الفئات
    probabilities = model.predict_proba([cleaned_text])[0]
    max_prob = max(probabilities)
    predicted_tag = model.classes_[probabilities.argmax()]
    
    # وضع حد أدنى لدرجة الثقة (مثلاً 40%)
    CONFIDENCE_THRESHOLD = 0.40
    
    if max_prob < CONFIDENCE_THRESHOLD:
        response = "عذراً، لم أفهم ما تقصده بوضوح. هل يمكنك إعادة الصياغة؟"
    else:
        response = get_response(predicted_tag)
    
    await cl.Message(content=response).send()
