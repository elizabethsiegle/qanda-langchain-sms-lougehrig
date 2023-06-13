import requests
from langchain.document_loaders import TextLoader

def loadTXTFileFromURL(text_file_url='https://raw.githubusercontent.com/elizabethsiegle/qanda-langchain-sms-lougehrig/main/lougehrig.txt'):
    # Fetching the text file
    output_file_name = "url_text_file.txt"
    response = requests.get(text_file_url)
    with open(output_file_name, "w",  encoding='utf-8') as file:
      file.write(response.text)

    # Load the text document using TextLoader
    loader = TextLoader('./'+output_file_name)
    loaded_docs = loader.load()
    return loaded_docs

from langchain.text_splitter import CharacterTextSplitter
def splitDocument(loaded_docs):
    # Splitting documents into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_docs = splitter.split_documents(loaded_docs)
    return chunked_docs
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
def createEmbeddings(chunked_docs):
    # Create embeddings and store them in a FAISS vector store
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunked_docs, embedder)
    return vector_store
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
def loadLLMModel():
    llm=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def askQuestions(vector_store, chain, question):
    # Ask a question using the QA chain
    similar_docs = vector_store.similarity_search(question)
    response = chain.run(input_documents=similar_docs, question=question)
    return response
chain = loadLLMModel()
LOCAL_loaded_docs = loadTXTFileFromURL()
LOCAL_chunked_docs = splitDocument(LOCAL_loaded_docs)
LOCAL_vector_store = createEmbeddings(LOCAL_chunked_docs)
LOCAL_response = askQuestions(LOCAL_vector_store, chain, "What does Lou Gehrig have to live for?")
print(LOCAL_response)