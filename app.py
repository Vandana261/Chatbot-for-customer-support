import json
import random

from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Load intents from JSON file
with open('intents.json', 'r') as file:
    intents = json.load(file)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            if pattern.lower() in text.lower():
                return random.choice(intent['responses'])

    # If no matching pattern is found, use the DialoGPT model
    # (Assuming 'tokenizer' and 'model' are defined globally)
    return generate_response_with_gpt(text)


def generate_response_with_gpt(text):
    new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')
    chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

if __name__ == '__main__':
    app.run()
