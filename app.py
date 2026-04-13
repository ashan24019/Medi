from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from src.prompt import *
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

index_name = "medi-chat"
embeddings = download_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chatModel = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}")
])

question_answering_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    print("User question:", msg)

    try:
        result = rag_chain.invoke({"input": msg})
    except TypeError:
        if hasattr(rag_chain, "run"):
            result = rag_chain.run(msg)
        else:
            raise

    answer = None
    if isinstance(result, dict):
        for key in ("answer", "output_text", "output", "result", "text", "response"):
            if key in result:
                answer = result[key]
                break
        if answer is None:
            if isinstance(result.get("outputs"), (list, tuple)) and result.get("outputs"):
                first = result["outputs"][0]
                if isinstance(first, dict):
                    for k in ("answer", "output_text", "text", "response"):
                        if k in first:
                            answer = first[k]
                            break
                if answer is None:
                    answer = str(first)
            else:
                answer = str(result)
    else:
        answer = str(result)

    print("Response:", answer)
    return answer



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)