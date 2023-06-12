from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import OpenAI

# load document
# pdf_path = "/Users/lsiegle/Downloads/langchaincookbookfundamentals.pdf"
# Using text-davinci-003 and a temperature of 0
# llm = OpenAI(model_name='text-davinci-003', temperature=0, openai_api_key=OPENAI_API_KEY)

import requests

text_url = 'https://raw.githubusercontent.com/hwchase17/chat-your-data/master/state_of_the_union.txt'
response = requests.get(text_url)

#let'extract only the text from the response
data = response.text
# print(data)

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(data)

embeddings = OpenAIEmbeddings()
persist_directory = 'db'
docsearch = Chroma.from_texts(
    texts, 
    embeddings,
    persist_directory = persist_directory,
    metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))]
    )
# Convert the vectorstore to a retriever
retriever = docsearch.as_retriever()
docs = retriever.get_relevant_documents("https://raw.githubusercontent.com/elizabethsiegle/qanda-langchain-sms-lougehrig/main/lougehrig.txt")

#create the chain to answer questions
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=OpenAI(temperature=0), 
    chain_type="stuff", 
    retriever=retriever,
    return_source_documents=True
)
def process_result(result):
    print(result['answer'])
    print("\n\n Sources : ", result['sources'])
    print(result['sources'])
question = "how much does lou gehrig have to live for"
result = chain({"question": question})
process_result(result)
