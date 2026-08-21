import os
from groq import AsyncGroq
import chainlit as cl

# تهيئة العميل باستعمال Async للسرعة والأداء
client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))

@cl.on_chat_start
async def start_chat():
    # إعداد السجل الأولي للمحادثة عند فتح المستخدم للصفحة
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "أنت مساعد ذكي ولطيف تتحدث باللغة العربية."}]
    )

@cl.on_message
async def main(message: cl.Message):
    # جلب سجل المحادثة السابق
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    # إعداد رسالة فارغة في الشاشة لاستقبال الرد المباشر (Streaming)
    msg = cl.Message(content="")
    await msg.send()

    # استدعاء نموذج Qwen عبر Groq بنفس أسلوب البث
    stream = await client.chat.completions.create(
        model="qwen-2.5-32b",
        messages=message_history,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices[0].delta.content:
            await msg.stream_token(chunk.choices[0].delta.content)

    # تحديث النتيجة النهائية وحفظها في السجل
    await msg.update()
    message_history.append({"role": "assistant", "content": msg.content})
    cl.user_session.set("message_history", message_history)
