import os
import re
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

llm_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# كلمات حظر البرمجة
CODING_KEYWORDS = [
    r"\bpython\b", r"\bjavascript\b", r"\bhtml\b", r"\bcss\b", r"\bjava\b", 
    r"\bc\+\+\b", r"\bphp\b", r"\bsql\b", r"\bcode\b", r"\bscript\b",
    r"اكتب كود", r"كود برلمجي", r"برمج لي", r"دالة", r"خوارزمية", r"تصحيح الكود",
    r"function\s*\(", r"def\s+\w+\(", r"import\s+\w+", r"class\s+\w+"
]

def is_coding_request(text: str) -> bool:
    text_lower = text.lower()
    for pattern in CODING_KEYWORDS:
        if re.search(pattern, text_lower):
            return True
    return False

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

embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)
vector_store = FAISS.load_local(
    FAISS_INDEX_PATH, 
    embeddings, 
    allow_dangerous_deserialization=True
)

# ==========================================
# 3. معالجة الرسائل
# ==========================================
@cl.on_message
async def main(message: cl.Message):
    # 1. تصفية الطلبات البرمجية فوراً
    if is_coding_request(message.content):
        await cl.Message(
            content="عذراً، هذا الشاتبوت مخصص فقط للإجابة عن استفسارات ومعاملات المديرية العامة لتربية نينوى ولا يقدم خدمات برمجية."
        ).send()
        return

    try:
        docs = await asyncio.to_thread(
            vector_store.similarity_search, message.content, k=2
        )

        context = "\n\n".join([d.page_content for d in docs]) if docs else "لا يوجد سياق متوفر."

        # 2. تعزيز التعليمات بعدم الإجابة عن البرمجة
        system_instruction = f"""أنت مساعد ذكي ومفيد مخصص لخدمة موظفي ومراجعي المديرية العامة لتربية نينوى. 

قواعد صارمة:
- أجب عن سؤال المستخدم باللغة العربية بناءً على السياق المرفق فقط.
- يمنع منعاً باتاً تقديم أي أكواد برمجية أو الإجابة عن أسئلة البرمجة والتطوير.

السياق المتاح:
{context}"""

        msg = cl.Message(content="")
        await msg.send()

        stream_response = await llm_client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct",
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

        async for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                await msg.stream_token(token)

        await msg.update()

    except Exception as e:
        await cl.Message(content=f"حدث خطأ أثناء معالجة الطلب: {str(e)}").send()
