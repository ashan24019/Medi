from dotenv import load_dotenv
import os
from src.helper import load_pdf, filter_source, text_spliter, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

file_path = "data/"

extracted_docs = load_pdf(file_path=file_path)
filtered_docs = filter_source(extracted_docs)
text_chunks = text_spliter(filtered_docs)

embeddings = download_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medi-chat"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embeddings,
    index_name = index_name
)


