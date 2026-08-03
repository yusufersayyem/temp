import chainlit as cl

@cl.on_chat_start
async def start():
    # إعلان في الشريط الجانبي عند بدء المحادثة
    await cl.Sidebar.set_title("إعلانات ورعاة")
    await cl.Sidebar.add_element(
        cl.Text(content="🌟 **برعاية شركة X**\nاحصل على خصم 20% بضغطة زر!")
    )
    
    await cl.Message(content="أهلاً بك! كيف يمكنني مساعدتك اليوم؟").send()

@cl.on_message
async def main(message: cl.Message):
    # رد الذكاء الاصطناعي العادي
    user_intent = message.content.lower()
    
    # محاكاة إرسال رد مع صورة إعلانية
    image = cl.Image(
        path="./ad_banner.png", # ضع مسار الصورة الإعلانية هنا
        name="عرض خاص!",
        display="inline" # لعرض الصورة مباشرة داخل المحادثة
    )
    
    # إضافة زر تفاعلي لشراء أو زيارة الرابط
    actions = [
        cl.Action(
            name="visit_offer", 
            value="https://example.com/offer", 
            label="🔗 استفد من العرض الآن"
        )
    ]

    await cl.Message(
        content="إليك إجابة سؤالك... \n\n📢 **إعلان مالي/ترويجي:** لا تفوت عرضنا الساري اليوم!",
        elements=[image],
        actions=actions
    ).send()

@cl.action_callback("visit_offer")
async def on_action(action: cl.Action):
    # عند النقر على الزر
    await cl.Message(content=f"شكرًا لاهتمامك! يمكنك زيارة العرض من هنا: {action.value}").send()
