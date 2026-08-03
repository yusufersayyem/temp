import chainlit as cl

@cl.on_message
async def main(message: cl.Message):
    # Echo back the user's message or add your AI model logic here
    response = f"You said: {message.content}"
    
    # Send the response back to the user
    await cl.Message(content=response).send()
