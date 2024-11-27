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
    try:
        structured_llm = llm.with_structured_output(Notes)
        
        prompt = f"""Create detailed notes in markdown format for the following content:

        Requirements:
        - Use markdown formatting (headings, bullet points, emphasis)
        - Include a clear title
        - Organize content logically
        - Keep explanations clear and concise

        Content to structure:
        {extracted_text_topic_wise}

        Format the response as a Notes object with a title and markdown-formatted content."""
        
        response = structured_llm.invoke(prompt)
        return response
        
    except Exception as e:
        print(f"Error generating notes: {str(e)}")
        # Return a default Notes object in case of error
        return Notes(
            title="Error Processing Content",
            content="Unable to generate structured notes. Please try again with different content."
        )