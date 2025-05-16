import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import faiss
import shutil
import markdown
import base64
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# Import utility modules
from utils.syllabus_extractor import syllabus_llm
from utils.text_embedder import improved_text_embedder
from utils.notes_outputter import improved_notes_maker
from utils.manim_generator import ManimGenerator
from utils.notes_handler import (
    clean_syllabus,
    calculate_similarity,
    enhanced_retriever,
    get_video_embed_html,
    consolidate_unit_notes_with_animations,
    display_notes_with_html,
    create_final_html
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Smart Study Notes Generator with Animations",
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

# Generate Manim animation for a topic
def generate_topic_animation(topic, concepts=None, manim_gen=None):
    if not manim_gen:
        manim_gen = ManimGenerator()
    
    if not manim_gen.ensure_environment():
        return False, "Manim not installed properly"
    
    try:
        success, result = manim_gen.create_animation(topic, concepts)
        return success, result
    except Exception as e:
        return False, f"Error generating animation: {str(e)}"

# Main application UI
st.title("📝 Smart Study Notes Generator with Animations")
st.write("Upload your study materials and provide your syllabus to generate comprehensive notes with animations!")

# File upload section
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

# Notes generation section
st.header("📋 Generate Notes from Syllabus")
st.write("Paste your syllabus below to generate comprehensive study notes with animations")

user_syllabus = st.text_area("Enter your syllabus here:", height=200)
generate_button = st.button("Generate Study Notes with Animations")

# Initialize the Manim generator
manim_gen = ManimGenerator()

# Check Manim installation
if generate_button:
    if not manim_gen.ensure_environment():
        st.error("Manim is not installed properly. Please install it to generate animations.")

# Generation status placeholder
generation_status = st.empty()

if generate_button and user_syllabus and manim_gen.ensure_environment():
    generation_status.info("Processing syllabus...")
    
    # Clean the syllabus
    cleaned_syllabus = clean_syllabus(user_syllabus)
    
    # Extract structured syllabus
    result = syllabus_llm(cleaned_syllabus)
    
    # Store syllabus in session state
    st.session_state.syllabus = result
    
    # Track all notes and chunks for deduplication
    all_notes = []
    all_chunks_used = []
    
    # Store topic notes
    st.session_state.topic_notes = {}
    
    # Create directory for notes assets
    os.makedirs("notes_assets", exist_ok=True)
    os.makedirs(os.path.join("notes_assets", "videos"), exist_ok=True)
    
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
        unit_animations = {}
        
        for topic_idx, topic in enumerate(unit.topics):
            progress_text.write(f"Generating notes for: {topic}")
            
            # Retrieve relevant chunks with enhanced retriever
            retrieved_chunks = enhanced_retriever(
                topic, 
                previous_chunks=all_chunks_used,
                vector_store=vector_store
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
            
            # Store topic notes
            st.session_state.topic_notes[topic] = topic_notes
            
            # Generate animation for this topic using extracted concepts
            progress_text.write(f"Generating animation for: {topic}")
            if hasattr(topic_notes, 'concepts') and topic_notes.concepts:
                success, result_animation = generate_topic_animation(topic, topic_notes.concepts, manim_gen)
            else:
                success, result_animation = generate_topic_animation(topic, None, manim_gen)
            
            if success:
                # Create HTML embed for the animation and store it
                video_html = get_video_embed_html(result_animation)
                unit_animations[topic] = video_html
            
            # Add to unit notes
            unit_notes.append(f"## {topic_notes.title}\n\n{topic_notes.content}")
            
            # Update progress
            topics_processed += 1
            progress_percentage = topics_processed / topics_total
            progress_bar.progress(progress_percentage)
        
        # Consolidate unit notes with animations for better flow
        consolidated_unit = consolidate_unit_notes_with_animations(
            unit.title, 
            unit_notes, 
            unit_animations, 
            consolidation_llm
        )
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
6. DO NOT MODIFY OR REMOVE ANY HTML CONTENT (especially video tags)
7. Add section introductions where helpful

Your output should be well-formatted markdown that forms a complete study guide with embedded animations."""

    try:
        final_notes = consolidation_llm.invoke(final_prompt).content
    except Exception as e:
        st.error(f"Error during final consolidation: {str(e)}")
        final_notes = "\n\n".join(all_notes)
    
    # Save the markdown notes
    with open("notes.md", "w", encoding="utf-8") as notes_file:
        notes_file.write(final_notes)
    
    # Convert and prepare for display in Streamlit
    html_content = create_final_html(final_notes)
    
    # Complete status and show results
    generation_status.success("Notes generation with animations complete!")
    progress_text.empty()
    progress_bar.empty()
    
    # Display the generated notes with videos in Streamlit
    st.subheader("Generated Study Notes with Animations")
    display_notes_with_html(final_notes, st)
    
    # Provide download button for HTML
    with open("notes.html", "w", encoding="utf-8") as html_file:
        html_file.write(html_content)
    
    with open("notes.html", "r", encoding="utf-8") as f:
        html_download = f.read()
        st.download_button("📥 Download Complete HTML with Animations", html_download, "study_notes.html")
    
    # Create a zip file of the notes_assets directory for download
    if os.path.exists("notes_assets"):
        # Create a zip file of the assets
        shutil.make_archive("notes_assets_zip", 'zip', "notes_assets")
        with open("notes_assets_zip.zip", "rb") as f:
            st.download_button("📥 Download Assets (Required for HTML)", f.read(), "notes_assets.zip")

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
    
    # Manim animation options
    st.subheader("🎬 Animation Settings")
    
    # Check if Manim is installed
    if not manim_gen.ensure_environment():
        st.error("Manim not installed. Animations will be skipped.")
        st.info("""
        To install Manim:
        
        For macOS:
        ```
        brew install py3cairo ffmpeg pango scipy
        pip install manim
        ```
        
        For other systems, see: https://docs.manim.community/en/stable/installation.html
        """)
    else:
        st.success("Manim installed successfully! Animations will be included.")
    
    # Help section
    st.subheader("📖 How to Use")
    st.markdown("""
    1. **Upload PDFs** with your learning materials
    2. Click **Process Uploaded Files**
    3. Paste your **syllabus** in the text area
    4. Click **Generate Study Notes with Animations**
    5. The app will automatically:
       - Generate comprehensive notes
       - Create animations for each topic
       - Embed animations within the notes
    6. Download the complete HTML notes with videos
    """)