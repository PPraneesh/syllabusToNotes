import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from syllabus_extractor import syllabus_llm
from text_embedder import text_embedder
from notes_outputter import notes_maker

load_dotenv()

# embeddings = CohereEmbeddings(
#     model="embed-english-v3.0",
# )

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
# vector_store =FAISS.load_local("vector_store",embeddings,allow_dangerous_deserialization=True)

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
for uploaded_file in uploaded_files:
    if uploaded_file is not None:
        # Ensure the 'data' directory exists
        os.makedirs("data", exist_ok=True)
        temp_file_path = os.path.join("data", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        status = text_embedder(temp_file_path,vector_store)
        vector_store.save_local("vector_store")

user_input = st.text_input("Enter your query here:")
if user_input :
    st.write(user_input)
    if True :
        retriever = vector_store.as_retriever(search_kwargs={"k": 20})
        compressor = CohereRerank(model="rerank-english-v3.0")
        compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor, base_retriever=retriever)
        results = retriever.invoke(user_input)
        if len(results) == 0:
            print("No results found")
            st.write("No results found")
        else:
            print(results)
            st.write(results)

reload_button_resp = st.button("Reload embeddings")
if reload_button_resp:
    vector_store =FAISS.load_local("vector_store",embeddings,allow_dangerous_deserialization=True)
user_syllabus = st.text_area("Enter your syllabus here:")
if user_syllabus:
    result = syllabus_llm(user_syllabus)
    notes = []
    
    for res in result.units:
        for topic in res.topics:
            st.write(topic)
            retriever = vector_store.as_retriever(search_kwargs={"k": 20})
            compressor = CohereRerank(model="rerank-english-v3.0")
            compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, base_retriever=retriever)
            results = compression_retriever.invoke(topic)
            if len(results) == 0:
                print("No results found")
                st.write("No results found")
            else:
                print(results)
                notes.append(notes_maker(results))
    st.write(notes)
    # Save notes to a text file
    with open("notes.md", "w",encoding="utf-8") as notes_file:
        for note in notes:
            notes_file.write(note.title + "\n")
            notes_file.write(note.content + "\n")
    st.success("Notes have been saved to notes.txt")
    # pass it to strealit to download the file
    st.markdown("### [Download the notes](notes.txt)")