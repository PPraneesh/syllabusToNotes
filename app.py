import os
import asyncio
import streamlit as st
from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from utils.syllabus_extractor import syllabus_llm
from utils.text_embedder import improved_text_embedder
from utils.notes_outputter import improved_notes_maker
import markdown
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from typing import List, Dict
import re

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Study Notes Generator",
    page_icon="📝",
    layout="wide"
)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004", 
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

# Initialize large LLM for consolidation
consolidation_llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY")
)

# Initialize vector store
@st.cache_resource
def initialize_vector_store():
    try:
        vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except:
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        return vector_store

vector_store = initialize_vector_store()

# Function to clean and process syllabus text
def clean_syllabus(syllabus_text):
    # Remove excessive new lines and spaces
    cleaned = re.sub(r'\n{3,}', '\n\n', syllabus_text)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    # Ensure proper markdown formatting
    cleaned = re.sub(r'\*\*UNIT-([IVX]+):\*\*', r'## UNIT-\1:', cleaned)
    return cleaned

# Function to calculate text similarity (simple implementation)
def calculate_similarity(text1, text2):
    # Convert to sets of words for a basic Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

# Function to enhance retrieval with deduplication and better ranking
def enhanced_retriever(topic, previous_chunks=None, k=20):
    # Convert previous chunks to text if provided
    previous_texts = [chunk.page_content for chunk in previous_chunks] if previous_chunks else []
    
    # Initial retrieval with more chunks than needed
    retriever = vector_store.as_retriever(search_kwargs={"k": k + 10})
    
    # Use Cohere's reranker for better relevance
    # compressor = CohereRerank(
    #     model="rerank-english-v3.0",
    #     top_n=k,
    # )
    
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor,
    #     base_retriever=retriever
    # )
    
    # Get ranked results
    results = retriever.invoke(topic)
    
    # Filter out similar chunks to avoid repetition
    if previous_texts:
        filtered_results = []
        for res in results:
            is_duplicate = False
            for prev_text in previous_texts:
                if calculate_similarity(res.page_content, prev_text) > 0.7:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_results.append(res)
        
        # If we filtered too many, add some back based on ranking
        if len(filtered_results) < k // 2:
            # Add back some high-ranked chunks even if slightly similar
            for res in results:
                if res not in filtered_results and len(filtered_results) < k // 2:
                    filtered_results.append(res)
                    
        return filtered_results
    
    return results

# Function to consolidate notes across a unit
def consolidate_unit_notes(unit_title, topic_notes_list):
    try:
        prompt = f"""You are an expert educational content creator. Consolidate the following topic notes into a cohesive unit of study.

Unit Title: {unit_title}

Topic Notes:
{topic_notes_list}

Requirements:
1. Create a comprehensive unit that flows naturally between topics
2. Eliminate any redundant or repeated information
3. Ensure all key concepts are covered thoroughly
4. Maintain consistent formatting and style
5. Create natural transitions between topics
6. Ensure comprehensive coverage of the entire unit
7. Use markdown formatting for the final output

Your output should be properly formatted markdown that presents this unit in a clear, cohesive manner."""

        response = consolidation_llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"# {unit_title}\n\n*Error during consolidation: {str(e)}*\n\n" + "\n\n".join(topic_notes_list)

# Streamlit UI
st.title("📝 Smart Study Notes Generator")
st.write("Upload your study materials and provide your syllabus to generate comprehensive notes!")


st.header("📚 Upload Learning Materials")
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

# Process button instead of automatic processing
process_files = st.button("Process Uploaded Files")

# Processing status placeholder
processing_status = st.empty()

# Process files when button is clicked
if process_files and uploaded_files:
    processing_status.info("Starting document processing...")
    
    for i, uploaded_file in enumerate(uploaded_files):
        if uploaded_file is not None:
            # Create directory for data
            os.makedirs("data", exist_ok=True)
            temp_file_path = os.path.join("data", uploaded_file.name)
            
            # Save file
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Update status
            processing_status.info(f"Processing file {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
            
            # Embed file using improved embedder
            improved_text_embedder(temp_file_path, vector_store)
    
    # Save vector store
    vector_store.save_local("vector_store")
    processing_status.success("All documents processed successfully!")


st.header("📋 Generate Notes from Syllabus")
st.write("Paste your syllabus below to generate comprehensive study notes")

user_syllabus = st.text_area("Enter your syllabus here:", height=200)
generate_button = st.button("Generate Study Notes")

# Generation status placeholder
generation_status = st.empty()

if generate_button and user_syllabus:
    generation_status.info("Processing syllabus...")
    
    # Clean the syllabus
    cleaned_syllabus = clean_syllabus(user_syllabus)
    
    # Extract structured syllabus
    result = syllabus_llm(cleaned_syllabus)
    
    # Track all notes and chunks for deduplication
    all_notes = []
    all_chunks_used = []
    
    # Create progress metrics
    units_total = len(result.units)
    topics_total = sum(len(unit.topics) for unit in result.units)
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    topics_processed = 0
    
    # Process each unit and topic
    for unit_idx, unit in enumerate(result.units):
        generation_status.info(f"Processing Unit {unit_idx+1}/{units_total}: {unit.title}")
        unit_notes = []
        
        for topic_idx, topic in enumerate(unit.topics):
            progress_text.write(f"Generating notes for: {topic}")
            
            # Retrieve relevant chunks with enhanced retriever
            retrieved_chunks = enhanced_retriever(
                topic, 
                previous_chunks=all_chunks_used,
                k=15  # Adjust based on topic complexity
            )
            
            if not retrieved_chunks:
                unit_notes.append(f"## {topic}\n\n*No relevant information found for this topic.*")
                continue
            
            # Track chunks for deduplication
            all_chunks_used.extend(retrieved_chunks)
            
            # Prepare retrieved texts
            retrieved_texts = "\n\n".join([chunk.page_content for chunk in retrieved_chunks])
            
            # Generate notes with improved generator
            previous_notes_content = "\n".join([note for note in unit_notes])
            topic_notes = improved_notes_maker(topic, retrieved_texts, previous_notes_content)
            
            # Add to unit notes
            unit_notes.append(f"## {topic_notes.title}\n\n{topic_notes.content}")
            
            # Update progress
            topics_processed += 1
            progress_percentage = topics_processed / topics_total
            progress_bar.progress(progress_percentage)
        
        # Consolidate unit notes for better flow
        consolidated_unit = consolidate_unit_notes(unit.title, unit_notes)
        all_notes.append(f"# {unit.title}\n\n{consolidated_unit}")
    
    generation_status.info("Creating final consolidated notes...")
    
    # Create final consolidated output with cross-references
    final_prompt = f"""You are an expert educational content creator. Review these study notes and create a cohesive, well-organized study guide.

Study Notes:
{''.join(all_notes)}

Requirements:
1. Create a comprehensive study guide with all the information
2. Add cross-references between related topics
3. Ensure proper hierarchical structure with appropriate headings
4. Add a table of contents at the beginning
5. Eliminate any remaining redundancies
6. Ensure all content is in proper markdown format
7. Add section introductions where helpful

Your output should be well-formatted markdown that forms a complete study guide."""

    try:
        final_notes = consolidation_llm.invoke(final_prompt).content
    except Exception as e:
        st.error(f"Error during final consolidation: {str(e)}")
        final_notes = "\n\n".join(all_notes)
    
    # Save notes to files
    with open("notes.md", "w", encoding="utf-8") as notes_file:
        notes_file.write(final_notes)
    
    # Convert to HTML
    html = markdown.markdown(final_notes, extensions=['tables', 'toc'])
    with open("notes.html", "w", encoding="utf-8") as notes_html:
        notes_html.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Study Notes</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; padding-bottom: 10px; }}
                h2 {{ color: #3498db; margin-top: 30px; }}
                h3 {{ color: #2980b9; }}
                code {{ background-color: #f8f8f8; padding: 2px 5px; border-radius: 3px; }}
                pre {{ background-color: #f8f8f8; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                blockquote {{ border-left: 4px solid #ccc; padding-left: 15px; color: #666; }}
                img {{ max-width: 100%; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            {html}
        </body>
        </html>
        """)
    
    # Complete status and show results
    generation_status.success("Notes generation complete!")
    progress_text.empty()
    progress_bar.empty()
    
    # Display the generated notes
    st.subheader("Generated Study Notes")
    st.markdown(html, unsafe_allow_html=True)
    
    # Provide download options
    col_md, col_html = st.columns(2)
    with col_md:
        st.download_button("📥 Download Markdown", final_notes, "study_notes.md")
    with col_html:
        with open("notes.html", "r", encoding="utf-8") as f:
            html_content = f.read()
            st.download_button("📥 Download HTML", html_content, "study_notes.html")

# Sidebar options
with st.sidebar:
    st.header("⚙️ Options")
    
    # Reload vector store button
    if st.button("🔄 Reload Vector Store"):
        try:
            vector_store = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
            st.success("Vector store reloaded successfully!")
        except Exception as e:
            st.error(f"Error reloading vector store: {str(e)}")
    
    # Advanced options
    st.subheader("Advanced Settings")
    
    chunk_size = st.slider("Chunk Size", min_value=500, max_value=5000, value=2000, step=500, 
                          help="Size of text chunks for document processing")
    
    chunk_overlap = st.slider("Chunk Overlap", min_value=50, max_value=500, value=200, step=50,
                             help="Overlap between consecutive chunks")
    
    similarity_threshold = st.slider("Similarity Threshold", min_value=0.5, max_value=0.9, value=0.7, step=0.05,
                                   help="Threshold for detecting similar content (higher = more strict)")
    
    st.info("Note: Changes to advanced settings will apply to newly processed documents.")
    
    # Help section
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Upload PDFs** in the left panel
    2. Click **Process Uploaded Files**
    3. Paste your **syllabus** in the right panel
    4. Click **Generate Study Notes**
    5. Download the generated notes as **Markdown** or **HTML**
    """)