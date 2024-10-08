import os
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Initialize the LLM
# llm = ChatCohere(model="command-r-plus")
os.environ.get('GROQ_API_KEY')
llm = ChatGroq(model="llama-3.1-70b-versatile")


# Define the Pydantic classes for structured output
class Topic(BaseModel):
    """Represents a single topic in the syllabus."""
    title: str = Field(description="The title of a single topic within a unit")


class Unit(BaseModel):
    """Represents a single unit in the syllabus, containing multiple topics."""
    title: str = Field(description="The title of the unit")
    topics: List[Topic] = Field(description="A list of topics (each containing one and only one topic title) that belong to the unit")


class Syllabus(BaseModel):
    """Represents the entire syllabus, structured into multiple units."""
    units: List[Unit] = Field(description="A list of units in the syllabus, each containing topics")


# Function to process the syllabus using structured output from the LLM
def syllabus_llm(syllabus_text: str):
    """
    This function will invoke the model to process and return structured syllabus data.
    """
    structured_llm = llm.with_structured_output(Syllabus)
    
    # Prompt the LLM with the raw text and ask it to structure it
    prompt = f"Divide the following syllabus into units and topics:\n\n{syllabus_text}"
    
    # Invoke the model with structured output
    response = structured_llm.invoke(prompt)
    
    return response