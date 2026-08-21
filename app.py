import os
from typing import List
import chainlit as cl
from huggingface_hub import AsyncInferenceClient, InferenceClient
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS

# 1. إعداد المتغيرات والمفاتيح
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-72B-Instruct"
EMBEDDING_MODEL_ID = "BAAI/bge-m3"
FAISS_INDEX_PATH = "faiss_index"

# 2. إنشاء فئة تخصيص لتوليد الـ Embeddings مباشرة لتجنب أخطاء الـ API القديمة
class DirectHFEmbeddings(Embeddings):
    def __init__(self, model_name: str, token: str):
        self.client = InferenceClient(model=model_name, token=token)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        response = self.client.feature_extraction(texts)
        return response.tolist() if hasattr(response, "tolist") else response

    def embed_query(self, text: str) -> List[float]:
        response = self.client.feature_extraction(text)
        # في حال إرجاع أبعاد متداخلة لسؤال واحد، يتم تسطيح القائمة
        if isinstance(response, list) and len(response) > 0 and isinstance(response[0], list):
            if isinstance(response[0][0], list):
                response = response[0][0]
            else:
                response = response[0]
        return response.tolist() if hasattr(response, "tolist") else response

# 3. تهيئة عميل Qwen وعميل الـ Embeddings
llm_client = AsyncInferenceClient(model=MODEL_ID, token=HF_TOKEN)
embeddings = DirectHFEmbeddings(model_name=EMBEDDING_MODEL_ID, token=HF_TOKEN)

# 4. تحميل قاعدة بيانات FAISS محلياً
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
        # البحث الدلالي في FAISS عن طريق الـ API الخارجي
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

        # إرسال المحادثة للنموذج عبر البث التدفقي (Streaming)
        stream = await llm_client.chat_completion(
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
        
        # تنظيف سياق المحادثة وتوفير الـ Tokens
        message_history[-1] = {"role": "user", "content": message.content}
        message_history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("message_history", message_history)

    except Exception as e:
        msg = cl.Message(content=f"حدث خطأ أثناء الاتصال بالنظام: {str(e)}")
        await msg.send()
