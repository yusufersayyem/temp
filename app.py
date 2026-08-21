import os
from dotenv import load_dotenv
import chainlit as cl

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_qdrant import QdrantVectorStore

# التعديل الهام: الاستيراد المباشر لتوافق الإصدارات الحديثة
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
COLLECTION_NAME = "my_pdf_documents"

@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="جاري الاتصال بالنموذج وقاعدة البيانات...")
    await msg.send()

    # 1. Embedding Model (BAAI/bge-m3)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 2. Qdrant Retriever
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3. LLM (Qwen 2.5 7B via Hugging Face Inference API)
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        temperature=0.2,
        max_new_tokens=512
    )

    # 4. System Prompt
    system_prompt = (
        "أنت مساعد ذكي ومؤدب متخصص في الإجابة على استفسارات تربية نينوى.\n"
        "استخدم المعلومات الواردة في السياق المرفق فقط للإجابة على سؤال المستخدم.\n"
        "إذا لم تجد الإجابة في السياق، قل بوضوح: 'عذراً، لا تتوفر هذه المعلومة ضمن بيانات تربية نينوى المتاحة حالياً.' ولا تقم بابتكار إجابات من عندك.\n\n"
        "السياق المتاح:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 5. Build RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # حفظ السلسلة في جلسة المستخدم
    cl.user_session.set("rag_chain", rag_chain)
    
    msg.content = "أهلاً بك! أنا مساعدك الذكي الخاص بتربية نينوى. كيف يمكنني مساعدتك اليوم؟"
    await msg.update()

@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("rag_chain")
    
    # إرسال طلب المعالجة واستلام الإجابة
    res = await rag_chain.ainvoke({"input": message.content})
    
    await cl.Message(content=res["answer"]).send()
