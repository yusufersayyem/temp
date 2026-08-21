import os
import chainlit as cl
from huggingface_hub import AsyncInferenceClient

# الحصول على رمز التوثيق (Token) الخاص بـ Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN")

# اختيار نموذج Qwen المتاح على Hugging Face
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"

client = AsyncInferenceClient(model=MODEL_ID, token=HF_TOKEN)

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{"role": "system", "content": "أنت مساعد ذكي ولطيف تتحدث باللغة العربية."}]
    )

@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    message_history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    try:
        # استدعاء نموذج Qwen عبر Hugging Face كبث مباشر (Streaming)
        stream = await client.chat_completion(
            messages=message_history,
            max_tokens=1024,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)

        await msg.update()
        message_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("message_history", message_history)

    except Exception as e:
        msg.content = f"حدث خطأ أثناء الاتصال بالنظام: {str(e)}"
        await msg.update()
