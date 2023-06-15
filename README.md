## Chat with a Document over SMS
This generative question-answering SMS chatbot reads a document containing Lou Gehrig's Farewell Speech using LangChain, Hugging Face, and Twilio in Python.
![sms example](https://github.com/elizabethsiegle/qanda-langchain-sms-lougehrig/assets/8932430/8972af5a-2576-4509-953d-0fd7c328fa4e)

### Prerequisites
- A Twilio account - [sign up for a free one here](https://www.twilio.com/try-twilio)
- A Twilio phone number with SMS capabilities - [learn how to buy a Twilio Phone Number here](https://support.twilio.com/hc/en-us/articles/223135247-How-to-Search-for-and-Buy-a-Twilio-Phone-Number-from-Console)
- HuggingFace Account – [make a HuggingFace Account here](https://huggingface.co/join)
- Python installed - [download Python here](https://www.python.org/downloads/)
- [ngrok](https://ngrok.com/download), a handy utility to connect the development version of our Python application running on your machine to a public URL that Twilio can access.

### Configuration
Since you will be installing some Python packages for this project, you will need to make a new project directory and a virtual environment.
If you're using a Unix or macOS system, open a terminal and enter the following commands:
```bash
mkdir lc-qa-sms 
cd lc-qa-sms 
python3 -m venv venv 
source venv/bin/activate 
!pip install langchain
!pip install requests
pip install flask
pip install faiss
pip install sentence-transformers
pip install twilio
pip install load_dotenv
```
If you're following this tutorial on Windows, enter the following commands in a command prompt window:
```bash
mkdir lc-qa-sms  
cd lc-qa-sms 
python -m venv venv 
venv\Scripts\activate 
pip install langchain
pip install requests
pip install flask
pip install faiss
pip install sentence-transformers
pip install twilio
pip install load_dotenv
```
[Get a Hugging Face Access Token here](https://huggingface.co/settings/tokens). On the command line in your root directory, run
```bash
export HUGGINGFACEHUB_API_TOKEN=replace-with-your-huggingfacehub-token
```

### Code
The meat of the code is in `app.py`. 
There are 5 helper functions:
1. `loadFileFromURL` writes from a URL to a local file, and then loads that file from the local file storage with LangChain's `TextLoader` library. Later the file will be passed over to the `split` method to create chunks.
2. `splitDoc` splits the document into chunks. This is important because LLMs can't process inputs that are too long. LangChain's `CharacterTextSplitter` function helps us do this–setting `chunk_size` to 1000 and `chunk_overlap` to 10 keeps the integrity of the file by avoiding splitting words in half.
3. `makeEmbeddings` converts the chunked document into embeddings (numerical representations of words) with Hugging Face and stores them in a FAISS Vector Store.
4. `askQs` conducts a similarity search to get the most semantically-similar documents to a given input which the LLM needs to best answer questions 
5. `loadLLM` defines and loads the Hugging Face Hub LLM to be used with your Access Token and starts the request on a similarity search embedded with some input question to the selected LLM, enabling a question-and-answer conversation.

This app uses the [Flan-Alpaca-Large](https://huggingface.co/declare-lab/flan-alpaca-large) LLM but you could use another model like [flan-t5-xl](https://huggingface.co/google/flan-t5-xl)


