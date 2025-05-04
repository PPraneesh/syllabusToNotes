import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize LLMs with different capabilities
llm_notes = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", temperature=0.2)

class Notes(BaseModel):
    """Enhanced representation of study notes with metadata"""
    title: str = Field(description="The title of the topic, exactly matching the syllabus topic")
    content: str = Field(description="The detailed content of the topic in markdown format")
    concepts: Optional[List[str]] = Field(description="List of key concepts covered in the notes", default=None)
    missing_concepts: Optional[List[str]] = Field(description="List of concepts that seem to be missing from retrieved content", default=None)

def improved_notes_maker(topic, extracted_text_topic_wise, previous_notes=None):
    """
    Enhanced notes generator that avoids repetition and ensures comprehensive coverage
    
    Args:
        topic (str): The topic to generate notes for
        extracted_text_topic_wise (str): The retrieved content for the topic
        previous_notes (str, optional): Previously generated notes to avoid repetition
        
    Returns:
        Notes: Structured notes object with title, content and metadata
    """
    try:
        structured_llm = llm_notes.with_structured_output(Notes)
        
        # Prepare prompt with comprehensive instructions
        prompt = f"""Your task is to create detailed, comprehensive study notes for the following topic:

TOPIC: {topic}

REQUIREMENTS:
1. Create notes that thoroughly cover all aspects of this topic
2. Use clear, educational language appropriate for college students
3. Use proper markdown formatting with:
   - Headings (use ## for main sections, ### for subsections)
   - Bullet points for lists 
   - *italic* or **bold** for emphasis
   - Code blocks for any technical content
   - Tables where appropriate
4. Structure content with logical flow and clear organization
5. Include examples, applications, or case studies where relevant
6. Ensure content is accurate, concise, and comprehensive
7. Avoid unnecessary repetition of information

CONTENT TO USE:
{extracted_text_topic_wise}

{'PREVIOUSLY COVERED CONTENT (AVOID REPEATING):\n' + previous_notes if previous_notes else ''}

IMPORTANT INSTRUCTIONS:
- The title MUST match exactly: "{topic}" (do not modify the topic name)
- If you identify important concepts that appear to be missing from the provided content, list them in the missing_concepts field
- Identify and list 3-7 key concepts covered in these notes in the concepts field
- The notes should be comprehensive but focused specifically on this topic
- If the topic seems to overlap with previous content, focus on unique aspects not previously covered

Format your response as a Notes object with title, content, concepts, and missing_concepts fields."""
        
        # Generate structured notes
        response = structured_llm.invoke(prompt)
        
        # Ensure notes have all required fields
        if not hasattr(response, 'concepts') or not response.concepts:
            response.concepts = []
        
        if not hasattr(response, 'missing_concepts') or not response.missing_concepts:
            response.missing_concepts = []
            
        return response
        
    except Exception as e:
        print(f"Error generating notes: {str(e)}")
        # Return a default Notes object in case of error
        return Notes(
            title=topic,
            content=f"# {topic}\n\nUnable to generate structured notes for this topic due to an error:\n\n```\n{str(e)}\n```\n\nPlease try again with different content.",
            concepts=[],
            missing_concepts=["Error occurred during processing"]
        )