import os
import chainlit as cl
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# تحميل متغيرات البيئة
load_dotenv()

# إعدادات Qdrant
QDRANT_PATH = "vectorstore/db_qdrant"
COLLECTION_NAME = "my_pdf_documents"


def build_rag_chain():
    # 1. إعداد نموذج التضمين وقاعدة البيانات
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    client = QdrantClient(path=QDRANT_PATH)

    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )

    # 2. إعداد النموذج والموجه (Prompt)
    llm = ChatMistralAI(
        model="mistral-large-latest", temperature=0.1, max_retries=2
    )

    system_prompt = (
        "أنت مساعد ذكي مخصص للإجابة عن استفسارات مديرية تربية نينوى وجامعة الموصل.\n"
        "استخدم المعلومات الواردة في السياق التالي فقط للإجابة على سؤال المستخدم.\n"
        "إذا لم تكن الإجابة موجودة في السياق، قل بوضوح أنك لا تملك المعلومة الرسمية حول ذلك.\n"
        "أجب باللغة العربية وبشكل دقيق ومختصر.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # 3. إنشاء سلسلة Retrieval Chain
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": 3}), combine_docs_chain
    )

    return retrieval_chain


@cl.on_chat_start
async def on_chat_start():
    """تنفذ هذه الدالة عند بدء محادثة جديدة"""
    # إرسال رسالة ترحيبية
    await cl.Message(
        content="أهلاً بك! أنا المجيب الآلي لمديرية تربية نينوى وجامعة الموصل. كيف يمكنني مساعدتك اليوم؟"
    ).send()

    # بناء الـ RAG Chain وحفظها في جلسة المستخدم (User Session)
    rag_chain = build_rag_chain()
    cl.user_session.set("rag_chain", rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    """تنفذ هذه الدالة عند استلام أي رسالة من المستخدم"""
    # استرجاع الـ RAG Chain الخاصة بالجلسة الحالية
    rag_chain = cl.user_session.get("rag_chain")

    # إنشاء كائن رسالة فارغ لطباعة الرد بشكل تدريجي (Streaming)
    res = cl.Message(content="")

    # استدعاء السلسلة وتمرير Callback Handler الخاص بـ Chainlit لتتبع الخطوات
    cb = cl.AsyncLangchainCallbackHandler()

    # تنفيذ الاستعلام
    response = await rag_chain.ainvoke(
        {"input": message.content}, config={"callbacks": [cb]}
    )

    # عرض النتيجة النهائية
    res.content = response["answer"]
    await res.send()
