import os
import asyncio
from typing import List
import chainlit as cl
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from openai import AsyncOpenAI

# 1. إعداد المتغيرات والمفاتيح
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# 2. تهيئة عميل OpenRouter (باستخدام مكتبة openai)
llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://localhost",  # يمكنك استبدالها برابط موقعك/تطبيقك إذا وجد
        "X-Title": "Arabic Qwen Chatbot",    # اسم تطبيقك لتصنيف الطلبات
    }
)

# 3. كلاس الـ Embeddings الخاص بـ HuggingFace
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

# 4. تحميل قاعدة البيانات المتجهة FAISS
embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)

vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# 5. معالجة رسائل المستخدم في Chainlit
@cl.on_message
async def main(message: cl.Message):
    try:
        # البحث في FAISS عن السياق المناسب (k=2 للحصول على معلومات كافية)
        docs = await asyncio.to_thread(
            vector_store.similarity_search, message.content, k=2
        )

        if docs:
            context = "\n\n".join([d.page_content for d in docs])
        else:
            context = "لا يوجد سياق متوفر في قاعدة البيانات."

        # صياغة التعليمات البرمجية الهيكلية (System Prompt) لتأطير النموذج
        system_instruction = f"""أنت مساعد ذكي ومفيد تتحدث باللغة العربية الفصحى الواضحة والودودة.
أجب على سؤال المستخدم بالاعتماد بشكل أساسي ومباشر على السياق المرفق أدناه.
إذا لم تجد الإجابة في السياق، أبلغ المستخدم بذلك بلباقة.

السياق المتاح:
{context}"""

        # إنشاء رسالة فارغة لبدء دفق الإجابة (Streaming)
        msg = cl.Message(content="")
        await msg.send()

        # إرسال الطلب لنموذج Qwen2.5 7B Instruct عبر OpenRouter
        stream_response = await llm_client.chat.completions.create(
            model="qwen/qwen-2.5-7b-instruct",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": message.content}
            ],
            temperature=0.3,    # ضبط درجة الابتكار لمنع التخيل أو الخروج عن السياق
            max_tokens=400,     # حد أقصى للرد لتقليل استهلاك التوكنز والحفاظ على السعر الأرخص
            stream=True
        )

        # استلام الإجابة وطباعة كل حرف/كلمة فور ظهورها
        async for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)

        # إنهاء البث وحفظ الرسالة الكاملة
        await msg.update()

    except Exception as e:
        await cl.Message(content=f"حدث خطأ أثناء الاتصال بالنظام: {str(e)}").send()
