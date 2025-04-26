import os
import sys
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from chatbot import *

# Generate API KEY from Claude and OpenAI website and define as a variable.
os.environ["ANTHROPIC_API_KEY"] =  "YOUR_API_KEYS"
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEYS"



LOCAL_PATH = "ข้อมูลยา 50 ชนิด.pdf" # Define the path to your local PDF file
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 0
LLM_MAX_TOKEN = 1000
LLM_MODEL = "claude-3-5-sonnet-20240620"
WORD_SEPARATOR = "\n\n"  

# Prompt Instruction
PROMPT_INSTRUCTION = """
        You are an AI thai language model assistant.
        You are an expert at answering questions about medicine.
        Answer the question based ONLY on the following context.


        {context}


        Original question: {question}"""



# Access the PDF file
DATA = get_pdf_file_path(LOCAL_PATH=LOCAL_PATH)
# Split the documents into chunks
TEXTS = chunk_splitter(DATA=DATA ,WORD_SEPARATOR=WORD_SEPARATOR, CHUNK_SIZE=CHUNK_SIZE, CHUNK_OVERLAP=CHUNK_OVERLAP)
# Word Embedding
DOCSEARCH = word_embedding(TEXTS=TEXTS)
# Customize the prompt template for the LLM
CUSTOM_TEMPLATE = prompt_instruction(PROMPT_INSTRUCTION=PROMPT_INSTRUCTION)




# Create the language model
llm1 = ChatAnthropic(model=LLM_MODEL, max_tokens=LLM_MAX_TOKEN)

# Create the ConversationalRetrievalChain with the custom prompt template
chain = ConversationalRetrievalChain.from_llm(
    llm=llm1,
    retriever=DOCSEARCH.as_retriever(),
    combine_docs_chain_kwargs={"prompt": CUSTOM_TEMPLATE}
)



# Chatbot loop
chat_history = []
query = None  # Initialize query to avoid potential reference error

while True:
    if not query:
        query = input("User: ")
    if query in ['quit', 'q', 'exit']:
        break
    result = chain.invoke({"question": query, "chat_history": chat_history})
    print("Chatbot:", result['answer'])

    chat_history.append((query, result['answer']))
    query = None # Reset query for the next iteration