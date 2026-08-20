import os
import chainlit as cl
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient(
    model="meta-llama/Llama-3.3-70B-Instruct",
    token=HF_TOKEN
)

@cl.on_chat_start
async def start():
    if not HF_TOKEN:
        await cl.Message(content="⚠️ يرجى إضافة HF_TOKEN في متغيرات البيئة!").send()
        return

    cl.user_session.set("message_history", [
        {"role": "system", "content": "أنت مساعد ذكي ومفيد تتحدث باللغة العربية دائماً."}
    ])
    await cl.Message(content="مرحباً بك! كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def main(message: cl.Message):
    history = cl.user_session.get("message_history", [])
    history.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")
    await msg.send()

    try:
        response = client.chat_completion(
            messages=history,
            max_tokens=1000,
            stream=True
        )

        full_response = ""
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)
                full_response += token

        await msg.update()
        history.append({"role": "assistant", "content": full_response})
        cl.user_session.set("message_history", history)

    except Exception as e:
        await cl.Message(content=f"⚠️ حدث خطأ أثناء الاتصال بالنموذج:\n`{str(e)}`").send()
