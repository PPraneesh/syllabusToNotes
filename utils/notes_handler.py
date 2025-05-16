import re
import os
import shutil
import markdown
import base64
import streamlit as st
from pathlib import Path

def clean_syllabus(syllabus_text):
    """
    Clean and process syllabus text for better formatting.
    
    Args:
        syllabus_text (str): Raw syllabus text
        
    Returns:
        str: Cleaned syllabus text with proper formatting
    """
    # Remove excessive new lines and spaces
    cleaned = re.sub(r'\n{3,}', '\n\n', syllabus_text)
    cleaned = re.sub(r' {2,}', ' ', cleaned)
    # Ensure proper markdown formatting
    cleaned = re.sub(r'\*\*UNIT-([IVX]+):\*\*', r'## UNIT-\1:', cleaned)
    return cleaned

def calculate_similarity(text1, text2):
    """
    Calculate text similarity using Jaccard similarity.
    
    Args:
        text1 (str): First text
        text2 (str): Second text
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Convert to sets of words for a basic Jaccard similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0

def enhanced_retriever(topic, previous_chunks=None, vector_store=None, k=20):
    """
    Enhanced retrieval with deduplication for better content selection.
    
    Args:
        topic (str): The topic to retrieve content for
        previous_chunks (list): Previously retrieved chunks to avoid duplication
        vector_store: Vector store for retrieval
        k (int): Number of chunks to retrieve
        
    Returns:
        list: Retrieved chunks with duplication minimized
    """
    # Convert previous chunks to text if provided
    previous_texts = [chunk.page_content for chunk in previous_chunks] if previous_chunks else []
    
    # Initial retrieval with more chunks than needed
    retriever = vector_store.as_retriever(search_kwargs={"k": k + 10})
    
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

def get_video_embed_html(video_path, width=640):
    """
    Generate HTML for embedding a video with correct paths.
    
    Args:
        video_path (str): Path to the video file
        width (int): Width of the video player
        
    Returns:
        str: HTML code for embedding the video
    """
    # Check if file exists
    if not os.path.exists(video_path):
        return f"<!-- Video not found: {video_path} -->"
    
    # Get filename and ensure notes_assets/videos exists
    video_filename = os.path.basename(video_path)
    target_dir = os.path.join("notes_assets", "videos")
    os.makedirs(target_dir, exist_ok=True)
    
    # Copy the video to ensure it's in the assets directory
    target_path = os.path.join(target_dir, video_filename)
    if video_path != target_path:  # Avoid copying if already in place
        shutil.copy2(video_path, target_path)
    
    # Use relative path for HTML export
    relative_path = f"videos/{video_filename}"
    
    # Create HTML with both a player tag and a download link
    video_html = f"""
    <div class="video-container">
        <video width="{width}" controls>
            <source src="{relative_path}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
        <p><a href="{relative_path}" download>Download this animation</a></p>
    </div>
    """
    return video_html

def consolidate_unit_notes_with_animations(unit_title, topic_notes_list, topic_animations, consolidation_llm):
    """
    Consolidate topic notes with animations into coherent unit notes.
    
    Args:
        unit_title (str): Title of the unit
        topic_notes_list (list): List of topic notes content
        topic_animations (dict): Dictionary mapping topics to animation HTML
        consolidation_llm: LLM for consolidation
        
    Returns:
        str: Consolidated unit notes with embedded animations
    """
    try:
        # Prepare content with animations
        enhanced_content = []
        for topic_content in topic_notes_list:
            # Extract topic name from content (assuming it starts with ## TopicName)
            topic_match = re.search(r'^## (.+?)$', topic_content, re.MULTILINE)
            if topic_match and topic_animations:
                topic_name = topic_match.group(1).strip()
                if topic_name in topic_animations:
                    # Insert animation after the topic header
                    header_end = topic_match.end()
                    animation_html = topic_animations[topic_name]
                    enhanced_content.append(
                        topic_content[:header_end] + 
                        "\n\n" + 
                        animation_html + 
                        "\n\n" + 
                        topic_content[header_end:]
                    )
                    continue
            
            # If no animation or no match, add original content
            enhanced_content.append(topic_content)
        
        # Now consolidate with the enhanced content
        prompt = f"""You are an expert educational content creator. Consolidate the following topic notes into a cohesive unit of study.

Unit Title: {unit_title}

Topic Notes:
{'\n\n'.join(enhanced_content)}

Requirements:
1. Create a comprehensive unit that flows naturally between topics
2. Eliminate any redundant or repeated information
3. Ensure all key concepts are covered thoroughly
4. Maintain consistent formatting and style
5. Create natural transitions between topics
6. Ensure comprehensive coverage of the entire unit
7. DO NOT MODIFY OR REMOVE ANY HTML CONTENT, ESPECIALLY VIDEO TAGS AND CONTAINERS
8. Use markdown formatting for the final output

Your output should be properly formatted markdown that presents this unit in a clear, cohesive manner."""

        response = consolidation_llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"# {unit_title}\n\n*Error during consolidation: {str(e)}*\n\n" + "\n\n".join(topic_notes_list)

def display_notes_with_html(markdown_content, st_instance):
    """
    Display markdown content with videos correctly rendered in Streamlit using HTML
    
    Args:
        markdown_content (str): Markdown content with HTML video tags
        st_instance: Streamlit instance for UI rendering
    """
    # Convert markdown to HTML
    html_content = markdown.markdown(markdown_content, extensions=['tables', 'toc'])
    
    # Create CSS for better styling
    css = """
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; 
        color:#fff;
        }
        h1 { color: #fff; border-bottom: 2px solid #eee; padding-bottom: 10px; }
        h2 { color: #fff; margin-top: 30px; }
        h3 { color: #fff; }
        code { background-color: #000; padding: 2px 5px; border-radius: 3px; }
        pre { background-color: #000; padding: 15px; border-radius: 5px; overflow-x: auto; }
        blockquote { border-left: 4px solid #ccc; padding-left: 15px; color: #666; }
        img { max-width: 100%; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #fff; padding: 8px; }
        tr:nth-child(even) { background-color: #000; }
        .video-container { margin: 20px 0; }
        video { display: block; max-width: 100%; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
    """
    
    # Modify video paths to work in Streamlit's local server context
    def convert_video_paths(html_content):
        # Find all video source paths and update them
        video_paths = re.finditer(r'<source src="(videos/[^"]+)"', html_content)
        
        for match in video_paths:
            rel_path = match.group(1)
            abs_path = os.path.join("notes_assets", rel_path)
            
            if os.path.exists(abs_path):
                # Read the video file and encode it as base64
                with open(abs_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    video_b64 = base64.b64encode(video_bytes).decode()
                    # Replace with a data URL
                    new_path = f"data:video/mp4;base64,{video_b64}"
                    html_content = html_content.replace(match.group(1), new_path)
        
        return html_content
    
    # Process the HTML content for videos
    processed_html = convert_video_paths(html_content)
    
    # Combine CSS and HTML content
    full_html = f"{css}\n{processed_html}"
    
    # Display HTML in Streamlit
    st_instance.components.v1.html(full_html, height=800, scrolling=True)

def create_final_html(markdown_content):
    """
    Create the final HTML document with proper styling and video embedding.
    
    Args:
        markdown_content (str): Final markdown content with video markers
        
    Returns:
        str: Complete HTML document with embedded videos
    """
    # Process markdown for HTML output (keep video tags)
    html_content = markdown.markdown(markdown_content, extensions=['tables', 'toc'])
    
    # Create full HTML document with proper styling
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Study Notes with Animations</title>
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
            .video-container {{ margin: 20px 0; }}
            video {{ display: block; max-width: 100%; }}
            a {{ color: #3498db; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    return full_html