import getpass
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

#load_dotenv does not load the env varaibles from os it self load them from the .env file 
load_dotenv() # This loads variables from .env into os.environ 
api_key = os.getenv("GROQ_API_KEY")
debug = os.getenv("DEBUG", "False") #if error happen searching for this env varaible nothing will show for secuarity reasons 

# Optional: Check if API key exists
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")


model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=500,
    timeout=30,
    max_retries=6,
)
#model that we will use to generate emmpeding for every token (bag of words)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#vector space that will have all our tokens emmeding that we will use latter for retrieval 
vector_store = InMemoryVectorStore(embeddings_model)

file_path = "books/Introduction to Algorithms, 3rd Edition - Thomas H. Cormen.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()



# Test the model 
str = "Hello! Tell me a short joke."
response = model.invoke(str)
print(response.content)
print(len(docs))