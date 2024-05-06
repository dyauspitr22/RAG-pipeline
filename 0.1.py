from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import chainlit as cl
from langchain.chains import ConversationChain
from langchain_community.llms import OpenAI
# from getpass import getpass

# Define the Hugging Face API token
HUGGINGFACE_API_TOKEN = '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'
# print(HUGGINGFACE_API_TOKEN)
# Define the repository ID of the Hugging Face model
repo_id = "tiiuae/falcon-7b-instruct"

# Initialize the Hugging Face endpoint
llm = HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 0.6, "max_new_tokens": 200})

# Define the prompt template
template = """
You are a helpful AI Assistant which answers the user's query in detail with factually correct information.  

{question}
"""
prompt = PromptTemplate(template = template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)

print(llm_chain.run("What is 2 plus 2?"))

@cl.on_chat_start
def setup_llm_chain():
    # Instantiate the chain for that user session
    prompt = PromptTemplate(template = template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=False)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def handle_message(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

    # Call the chain asynchronously
    res = await llm_chain.acall(message.content, callbacks=[cl.AsyncLangchainCallbackHandler()])
    # Do any post processing here
    # Send the response
    await cl.Message(content=res["text"]).send()



