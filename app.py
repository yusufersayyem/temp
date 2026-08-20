import os
import chainlit as cl
from groq import Groq

# جلب المفتاح من متغيرات البيئة
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@cl.on_chat_start
async def start():
    # تعليمات البداية للنموذج لضمان التحدث بالعربية
    cl.user_session.set("message_history", [
        {"role": "system", "content": "أنت مساعد ذكي ومفيد تتحدث باللغة العربية دائماً وبأسلوب لطيف."}
    ])
    await cl.Message(content="مرحباً بك! كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    history.append({"role": "user", "content": message.content})

    # إرسال طلب للنموذج
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # نموذج لاما مجاني وسريع
        messages=history,
        temperature=0.7,
    )

    answer = response.choices[0].message.content
    history.append({"role": "assistant", "content": answer})
    
    # عرض الإجابة في الواجهة
    await cl.Message(content=answer).send()
