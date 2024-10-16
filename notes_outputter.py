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
    content: str = Field(description="The detailed content of the topic")

def notes_maker(extracted_text_topic_wise: str):
    """
    This function will invoke the model to process and return structured, detailed notes.
    """
    structured_llm = llm.with_structured_output(Notes)
    
    # Updated prompt for more detailed and extensive notes
    prompt = f"""Create comprehensive and detailed structured notes for exam preparation based on the following content. For each distinct topic:

    1. Provide a clear and specific title that accurately represents the topic.
    2. Summarize the key points in the content, ensuring that EVERY point from the original text is included.
    3. Elaborate extensively on each point, providing thorough explanations, examples, and contextual information where applicable.
    4. Include any relevant sub-points, ensuring a hierarchical structure in the notes if necessary.
    5. If there are any concepts that require further clarification, provide that additional information.
    6. Ensure that the notes are comprehensive enough that a student could use them as a primary study resource.

    Rules for creating the notes:
    - Be thorough and expansive in your explanations. Aim for depth and breadth in covering each topic.
    - Do not summarize or condense information. Instead, expand on each point to provide a full understanding.
    - Use clear, concise language, but don't sacrifice detail for brevity.
    - Include all information from the original text, even if it seems redundant.
    - If the original text is brief, expand on the topics using general knowledge about the subject.
    - Format the notes in a way that's easy to read and study from, using bullet points, numbering, or paragraphs as appropriate.

    Here's the content to structure into detailed notes:
    {extracted_text_topic_wise}

    Please format the output as a Notes object with 'title' and 'content' fields, where the content is extensive and covers all points in great detail."""
    
    # Invoke the model with structured output
    response = structured_llm.invoke(prompt)
    
    return response