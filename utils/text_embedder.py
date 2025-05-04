import asyncio
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

async def load_pages(loader):
    """Load PDF pages asynchronously"""
    pages = []
    async for page in loader.alazy_load():
        # Add page number to metadata
        page.metadata["page_number"] = len(pages) + 1
        pages.append(page)
    return pages

def extract_headers(text):
    """Extract potential headers from text"""
    lines = text.split('\n')
    headers = []
    
    for line in lines:
        # Check for various header patterns
        if re.match(r'^#+\s+', line):  # Markdown headers
            headers.append(line.strip())
        elif re.match(r'^[A-Z\s]{5,}$', line):  # ALL CAPS lines
            headers.append(line.strip())
        elif '**' in line:  # Bold text
            headers.append(line.strip())
        elif re.match(r'^[\d\.]+\s+[A-Z]', line):  # Numbered sections starting with capital
            headers.append(line.strip())
    
    return headers

def improved_text_embedder(uploaded_file, vector_store):
    """
    Enhanced PDF loader and embedder with intelligent chunking
    
    Args:
        uploaded_file (str): Path to the uploaded PDF file
        vector_store: Vector store to add documents to
        
    Returns:
        bool: Success status
    """
    # Load PDF
    loader = PyPDFLoader(uploaded_file)
    pages = asyncio.run(load_pages(loader))
    
    # Extract document name for metadata
    document_name = uploaded_file.split('/')[-1]
    
    # Use a hybrid splitter strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Smaller chunks for more precise retrieval
        chunk_overlap=200,  # Larger overlap to maintain context
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],  # Respect document structure
        length_function=len,
    )
    
    # Process and enrich documents with metadata
    documents = text_splitter.split_documents(pages)
    
    # Track headers to identify sections
    current_section = ""
    
    # Enrich chunks with metadata
    for i, doc in enumerate(documents):
        # Basic metadata
        doc.metadata["chunk_id"] = i
        doc.metadata["document_name"] = document_name
        
        # Extract potential section headers
        headers = extract_headers(doc.page_content)
        if headers:
            # Add first header as potential section title
            current_section = headers[0]
        
        # Add section info to metadata
        doc.metadata["section"] = current_section
        doc.metadata["possible_headers"] = headers
        
        # Count tokens (approximation)
        doc.metadata["token_count"] = len(doc.page_content.split())
    
    # Add to vector store
    vector_store.add_documents(documents)
    
    return True