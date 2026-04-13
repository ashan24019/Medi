from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document


def load_pdf(file_path):
    loader = DirectoryLoader(
        file_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


def  filter_source( docs: List[Document]) -> List[Document]:
    filtered_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        filtered_docs.append(
            Document(
                page_content=doc.page_content,
                metadata= {"source": src}
            )
        )
    return filtered_docs


def text_spliter(filtered_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 20
    )

    texts_chunk = text_splitter.split_documents(filtered_docs)
    return texts_chunk


def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name
    )
    return embeddings
