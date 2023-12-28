from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import dotenv
import os

# Load API key from .env file
dotenv.load_dotenv()

# Load the content from "bot.txt"
with open("bot.txt", "r", encoding="utf-8") as bot_file:
    bot_text = bot_file.read()

# Create the vector storage
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts([bot_text], embeddings)

# Create the conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Setup the prompt template
system_template = """Use the following pieces of context to answer the user's question. Only use the supplied context to answer.
If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
qa_prompt = ChatPromptTemplate.from_messages(messages)

# Create the question and answer conversation chain
qa = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0), 
    vectorstore.as_retriever(search_kwargs={"k": 1}),  # Adjusted k value to 1
    combine_docs_chain_kwargs={"prompt": qa_prompt}, 
    memory=memory
)

while True:
    query = input("> ")
    if query == "exit":
        break
    result = qa({"question": query})
    print(result['answer'])
