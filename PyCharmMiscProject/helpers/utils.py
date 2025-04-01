import pandas as pd


def parse_whatsapp_chat(file):
    content = file.read().decode("utf-8")
    messages = content.split("\n")

    data = []
    for msg in messages:
        parts = msg.split(" - ", 1)
        if len(parts) == 2:
            timestamp, message = parts
            user_msg = message.split(": ", 1)
            if len(user_msg) == 2:
                user, msg_text = user_msg
                data.append({"Timestamp": timestamp, "User": user, "Message": msg_text})

    return pd.DataFrame(data)
