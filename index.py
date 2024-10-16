import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_cohere import CohereEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch

from syllabus_extractor import syllabus_llm

# # Load the environment variables
load_dotenv()
# os.environ.get('COHERE_API_KEY')

# Initialize the MongoDB client
client = MongoClient(os.environ.get("MONGODB_ATLAS_CLUSTER_URI"))
DB_NAME = "syllabusser_langchain"
COLLECTION_NAME = "syllabusser_vectorstores"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "syllabusser-index-vectorstores"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

# # Initialize the document loader
# loader = PyPDFLoader('./data/CNN.pdf')

# async def load_pages(loader):
#     pages = []
#     text = ""
#     async for page in loader.alazy_load():
#         pages.append(page)
#     return pages

# pages = asyncio.run(load_pages(loader))

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1024,
#     chunk_overlap= 100,
# )
# documents = text_splitter.split_documents(pages)

# # embedding model, here we used CohereEmbeddings
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
)

# Initialize the vector store
vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding =embeddings,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    relevance_score_fn="cosine"
)

# add documents to the vector store and create the vector search index

# def add_documents_to_vector_store(documents): 
#     """
#         This uses vector_store (which is an instance of MongoDBAtlasVectorSearch) to add documents to the vector store and create the vector search index (this is the index that is used to search for similar documents).
#     """
#     vector_store.add_documents(documents)
#     # this created the vector search index, without this you need to manually create the index using mongodb atlas UI, and this below line isn't mentioned in the documentation, try checking in api_reference instead of guides
#     vector_store.create_vector_search_index(
#     dimensions=1024
#     )

from text_embedder import text_embedder
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# add_documents_to_vector_store(documents)
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = os.path.join("data", uploaded_file.name)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Use the temporary file path with PyPDFLoader
    embedding_button_resp = st.button("Add embeddings")

    if embedding_button_resp:
        status = text_embedder(temp_file_path)
    # use st to take input from user and store it in a variable
    user_input = st.text_input("Enter your query here:")
    if user_input :
        st.write(user_input)
        if True :
            results = vector_store.similarity_search(
                user_input, k=2
            )
            if len(results) == 0:
                print("No results found")
                st.write("No results found")
            else:
                for res in results:
                    print(f"* {res.page_content} [{res.metadata}]")
                    st.write(f"* {res.page_content} [{res.metadata}]")



# result  = syllabus_llm("""UNIT-I:
# Formal Languages and Regular Expressions: Definition of Languages, Finite Automata
# - DFA, NFA, regular expressions, Conversion of regular expression to NFA, NFA to DFA,
# Pumping Lemma for regular languages, lex tools.
# UNIT-II:
# Overview of Compilation: Phases of Compilation - Lexical Analysis, Pass and Phases
# of translation, interpretation, bootstrapping, data structures in compilation.
# Context-free Grammars and Parsing: Context free grammars, derivation, parse trees,
# ambiguity, LL(K) grammars and LL (1) parsing, bottom-up parsing, handle pruning, LR
# Grammar Parsing, LALR parsing, YACC programming specification.
# UNIT-III:
# Semantics: Syntax directed translation, S-attributed and L-attributed grammars,
# Intermediate code - abstract syntax tree, translation of simple Assignments
# statements and control flow statements.
# UNIT-IV:
# Run Time Environments: Storage organization, storage allocation strategies, access to
# non-local names, language facilities for dynamics storage allocation.
# Code Optimization: Principal sources of optimization, Optimization of basic blocks,
# peephole optimization, flow graphs, optimization techniques.
# UNIT-V:
# Code Generation: Machine dependent code generation, object code forms, generic
# code generation algorithm, Register allocation and assignment. Using DAG
# representation of Block.
# """)

# print(result)