import os
import asyncio
from typing import List
import chainlit as cl
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from openai import AsyncOpenAI

# ==========================================
# 1. إعداد المتغيرات والمفاتيح
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# تهيئة عميل OpenRouter (AsyncOpenAI)
llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ==========================================
# 2. كلاس الـ Embeddings الخاص بـ HuggingFace
# ==========================================
class DirectHFEmbeddings(Embeddings):
    def __init__(self, model_name: str, token: str):
        self.client = InferenceClient(model=model_name, token=token)

    def _process_response(self, response) -> List[float]:
        if hasattr(response, "tolist"):
            response = response.tolist()

        while isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
            if isinstance(response[0][0], list):
                response = response[0]
            else:
                break

        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
            response = [sum(col) / len(response) for col in zip(*response)]

        return response

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        response = self.client.feature_extraction(text)
        return self._process_response(response)

# تحميل قاعدة بيانات FAISS عند بدء التشغيل
embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# ==========================================
# 3. معالجة الرسائل واستدعاء النموذج
# ==========================================
@cl.on_message
async def main(message: cl.Message):
    try:
        # البحث عن أكثر مستندين مرتبطين بسؤال المستخدم في FAISS
        docs = await asyncio.to_thread(
            vector_store.similarity_search, message.content, k=2
        )

        context = "\n\n".join([d.page_content for d in docs]) if docs else "لا يوجد سياق متوفر."

        # تعليمات النظام الموجهة للنموذج
        system_instruction = f"""أنت مساعد ذكي ومفيد مخصص لخدمة موظفي ومراجعي المديرية العامة لتربية نينوى. 
أجب عن سؤال المستخدم باللغة العربية بأسلوب واضح ومباشر بناءً على السياق المرفق أدناه فقط.

السياق المتاح:
{context}"""

        msg = cl.Message(content="")
        await msg.send()

        # إرسال الطلب لنموذج IBM Granite 4.0 Micro
        stream_response = await llm_client.chat.completions.create(
            model="ibm-granite/granite-4.0-h-micro",
            extra_headers={
                "HTTP-Referer": "https://localhost",
                "X-Title": "Nineveh Edu Chatbot",
            },
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": message.content}
            ],
            stream=True
        )

        # بث الإجابة تدريجياً للمستخدم
        async for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)

        await msg.update()

    except Exception as e:
        await cl.Message(content=f"حدث خطأ أثناء معالجة الطلب: {str(e)}").send()
