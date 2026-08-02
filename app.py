import chainlit as cl

# تظهر هذه الدالة عند فتح المستخدم للواجهة أول مرة
@cl.on_chat_start
async def start():
    # إرسال رسالة ترحيبية بداية المحادثة
    await cl.Message(
        content="أهلاً بك! 🤖 أنا المساعد الآلي، كيف يمكنني مساعدتك اليوم؟"
    ).send()

# تُستدعى هذه الدالة في كل مرة يرسل فيها المستخدم رسالة
@cl.on_message
async def main(message: cl.Message):
    # هنا يمكنك معالجة نص المستخدم من message.content
    user_input = message.content

    # رد تجريبي بسيط (يمكنك هنا ربطه مع الباك إند أو الـ Vector Store)
    bot_response = f"لقد استلمت سؤالك: '{user_input}'. جاري العمل على الرد..."

    # إرسال الإجابة للمستخدم
    await cl.Message(
        content=bot_response
    ).send()
