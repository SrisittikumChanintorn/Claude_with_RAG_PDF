import os 
import sys
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic

# Access the PDF file 
def get_pdf_file_path(LOCAL_PATH):
    """
    Function to get the PDF file path from the user.
    """
    if LOCAL_PATH:
        loader = PyMuPDFLoader(file_path=LOCAL_PATH)
        DATA = loader.load_and_split()
    else:
        print("Upload a PDF file")
        sys.exit()

    return DATA

# Chunk Splitter
def chunk_splitter(DATA ,WORD_SEPARATOR, CHUNK_SIZE, CHUNK_OVERLAP):
    """
    Function to Split the documents into chunks
    """
    text_splitter = CharacterTextSplitter(separator=WORD_SEPARATOR, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    TEXTS = text_splitter.split_documents(DATA)

    return TEXTS

# Word Embedding 
def word_embedding(TEXTS):
    """
    This part is used for embedding the docs and storing them into VectorDB and initializing the retriever.
    """
    embeddings = OpenAIEmbeddings()
    DOCSEARCH = Chroma.from_documents(TEXTS, embeddings)

    return DOCSEARCH 

# Customize the prompt template for the LLM
def prompt_instruction(PROMPT_INSTRUCTION):
    """
    This part is used for customize the prompt template for the LLM.
    """
    CUSTOM_TEMPLATE = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_INSTRUCTION,
    )

    return CUSTOM_TEMPLATE