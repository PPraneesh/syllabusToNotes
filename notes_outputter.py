import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

os.environ.get('GROQ_API_KEY')
llm = ChatGroq(model="llama-3.1-70b-versatile")

class Notes(BaseModel):
    """A class to represent detailed notes"""
    title: str = Field(description="The title of the topic")
    content: str = Field(description="The detailed content of the topic in markdown format")

def notes_maker(extracted_text_topic_wise: str):
    """
    This function will invoke the model to process and return structured, detailed notes in markdown format.
    """
    structured_llm = llm.with_structured_output(Notes)
    
    prompt = f"""Create comprehensive and detailed structured notes in markdown format for exam preparation based on the following content. Format the notes following these specific markdown rules:

    Formatting Requirements:
    1. Main topic title should be an H1 heading (#)
    2. Each major section should be an H2 heading (##)
    3. Use bullet points (-) for main points
    4. Use bold (**text**) for subtopics and key terms
    5. Add horizontal rules (---) after each major section
    6. Use proper indentation for nested points (2 spaces)
    7. Use italics (*text*) for definitions or important phrases
    8. Use code blocks (```) for any technical content or examples
    9. Create clear hierarchical structure with proper spacing
    10. Add a table of contents at the beginning if multiple sections exist

    Content Requirements:
    1. Provide clear and specific section headings
    2. Include ALL information from the original text
    3. Expand on each point with thorough explanations
    4. Add relevant examples and contextual information
    5. Include any necessary clarifications
    6. Make the notes comprehensive enough for self-study

    Example Format:
    # Main Topic Title

    ## First Major Section
    - **Subtopic 1**
      - Detailed explanation
      - Key points
    - **Subtopic 2**
      - Further details
      - Important concepts
    
    ---

    ## Second Major Section
    [continue format...]

    Here's the content to structure into detailed notes:
    {extracted_text_topic_wise}

    Please format the output as a Notes object with 'title' and 'content' fields, where the content follows the markdown formatting rules specified above and includes comprehensive information."""
    
    response = structured_llm.invoke(prompt)
    
    return response