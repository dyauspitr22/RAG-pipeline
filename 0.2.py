from flask import Flask, render_template, request, jsonify
from utils import SystemMessage, AIMessage, HumanMessage, ChatChain, ChatModel
from huggingface_hub import InferenceClient

app = Flask(__name__)

# Login to Hugging Face Hub
from huggingface_hub import login
login(token='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')

# Create Inference Client
client = InferenceClient()

# Initialize ChatModel
chat_model = ChatModel(client=client)

# Initialize Chat Chain
chat_chain = ChatChain([SystemMessage('You are a helpful AI that always gives right answers')])

# Function to interact with the chatbot
def interact_with_chatbot(message, chat_chain):
    # Add user message to chat chain
    chat_chain.chain.append(HumanMessage(message))
    
    # Invoke the chatbot
    output = chat_model.invoke(message, stream=True)
    
    # Add AI response to chat chain
    chat_chain.chain.append(AIMessage(output))
    
    return output

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['user_message']
    ai_response = interact_with_chatbot(user_message, chat_chain)
    return jsonify({'response': ai_response})

if __name__ == '__main__':
    app.run(debug=True)
