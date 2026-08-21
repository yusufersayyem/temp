import os
import asyncio
from dotenv import load_dotenv
import chainlit as cl

from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "my_pdf_documents"

# 1. إعداد FastEmbed لنموذج BAAI/bge-m3 بحجم ذاكرة ضئيل جدًا
print("جاري تحميل نموذج FastEmbed لـ BAAI/bge-m3...")
embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-m3")

# 2. ربط قاعدة البيانات Qdrant بالوسيط الصحيح (embedding)
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 3. إعداد النموذج اللغوي السريع من Groq
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 4. بناء الـ Prompt والـ Chain
template = """أنت مساعد ذكي ومؤدب متخصص في الإجابة على استفسارات تربية نينوى.
استخدم المعلومات الواردة في السياق المرفق فقط للإجابة على سؤال المستخدم.
إذا لم تجد الإجابة في السياق، قل بوضوح: 'عذراً، لا تتوفر هذه المعلومة ضمن بيانات تربية نينوى المتاحة حالياً.' ولا تقم بابتكار إجابات من عندك.

السياق المتاح:
{context}

السؤال: {question}
الإجابة:"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="أهلاً بك! أنا مساعدك الذكي الخاص بتربية نينوى. كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    def run_chain():
        return rag_chain.invoke(message.content)

    try:
        response_text = await asyncio.to_thread(run_chain)
        msg.content = response_text
        await msg.update()
    except Exception as e:
        msg.content = f"حدث خطأ أثناء معالجة الطلب: {str(e)}"
        await msg.update()
