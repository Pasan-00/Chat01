from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings, load_pdf_file, text_split
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os       

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
HuggingFaceHub_API_KEY = os.getenv("HUGGINGFACEHUB_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["HUGGINGFACEHUB_API_KEY"] = HuggingFaceHub_API_KEY

embeddings = download_hugging_face_embeddings()

index_name = "testv2"

docsearch = PineconeVectorStore.from_existing_index(
    index_name="testv2",
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = HuggingFaceHub(
    repo_id="facebook/bart-large-cnn",
    model_kwargs={"temperature": 0.1, "max_length": 500},
    )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

question_answering_chain = create_stuff_documents_chain(llm,prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response:", response['answer'])
    return str(response['answer'])


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000 , debug=True)