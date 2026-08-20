import os
import chainlit as cl
from groq import Groq

# جلب المفتاح
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

@cl.on_chat_start
async def start():
    # التأكد من وجود المفتاح
    if not GROQ_API_KEY:
        await cl.Message(content="⚠️ خطأ: لم يتم العثور على GROQ_API_KEY. يرجى التأكد من إضافته في Environment Variables على Render.").send()
        return

    # تهيئة السجل
    cl.user_session.set("message_history", [
        {"role": "system", "content": "أنت مساعد ذكي ومفيد تتحدث باللغة العربية دائماً وبأسلوب واضح وبسيط."}
    ])
    await cl.Message(content="مرحباً بك! كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history")
    
    if not history:
        history = [{"role": "system", "content": "أنت مساعد ذكي ومفيد تتحدث باللغة العربية دائماً."}]

    history.append({"role": "user", "content": message.content})

    # رسالة فارغة نضخ فيها الرد تدريجياً
    msg = cl.Message(content="")
    await msg.send()

    try:
        # طلب استجابة بتقنية Streaming
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=history,
            temperature=0.7,
            stream=True
        )

        full_response = ""
        for chunk in completion:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)
                full_response += token

        # تحديث الرسالة بالكامل وإضافتها للسجل
        await msg.update()
        history.append({"role": "assistant", "content": full_response})
        cl.user_session.set("message_history", history)

    except Exception as e:
        await cl.Message(content=f"حدث خطأ أثناء الاتصال بالنموذج: {str(e)}").send()
