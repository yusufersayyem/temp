import os
import chainlit as cl
from huggingface_hub import AsyncInferenceClient
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# 1. جلب المفاتيح والإعدادات
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# عميل Hugging Face لنموذج التوليد Qwen
client = AsyncInferenceClient(model=MODEL_ID, token=HF_TOKEN)

# 2. استدعاء نموذج التضمين عبر الـ API (بدون استهلاك RAM على Render)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name=EMBEDDING_MODEL_ID
)

# 3. تحميل فهرس FAISS محلياً مع ربطه بـ API التضمين
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

@cl.on_chat_start
async def start_chat():
    cl.user_session.set(
        "message_history",
        [{
            "role": "system", 
            "content": "أنت مساعد ذكي ومفيد. اعتمِد على السياق المرفق للإجابة عن أسئلة المستخدم بوضوح ودقة. إذا لم تجد الإجابة في السياق، أجب بما تعرفه بشكل عام."
        }]
    )

@cl.on_message
async def main(message: cl.Message):
    message_history = cl.user_session.get("message_history")

    try:
        # البحث الدلالي: يرسل السؤال لـ Hugging Face API للـ Embeddings ثم يبحث في FAISS
        docs = vector_store.similarity_search(message.content, k=2)
        
        context_text = ""
        if docs:
            context_text = "\n\n".join([doc.page_content for doc in docs])

        # دمج السياق مع سؤال المستخدم
        if context_text:
            user_prompt = f"المعلومات المستخرجة من قاعدة البيانات:\n{context_text}\n\nسؤال المستخدم: {message.content}"
        else:
            user_prompt = message.content

        message_history.append({"role": "user", "content": user_prompt})

        msg = cl.Message(content="")
        await msg.send()

        # استدعاء نموذج Qwen للوليد النصي عبر API
        stream = await client.chat_completion(
            messages=message_history,
            max_tokens=2048,
            temperature=0.3,
            stream=True
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)

        await msg.update()
        
        # تنظيف سجل المحادثة لتوفير الـ Tokens
        message_history[-1] = {"role": "user", "content": message.content}
        message_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("message_history", message_history)

    except Exception as e:
        msg = cl.Message(content=f"حدث خطأ أثناء الاتصال بالنظام: {str(e)}")
        await msg.send()
