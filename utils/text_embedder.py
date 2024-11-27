import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

async def load_pages(loader):
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

def text_embedder(uploaded_file,vector_store):
    loader = PyPDFLoader(uploaded_file)
    pages = asyncio.run(load_pages(loader))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap= 100,
    )
    documents = text_splitter.split_documents(pages)
    vector_store.add_documents(documents)
    return True