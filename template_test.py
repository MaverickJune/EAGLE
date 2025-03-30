from fastchat.model import get_conversation_template

your_message = "write a short story about a fox and a rabbit"
conv = get_conversation_template("vicuna")
conv.append_message(conv.roles[0], your_message)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

print(prompt)