import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
os.environ.get('GROQ_API_KEY')
llm = ChatGroq(model="llama-3.1-70b-versatile")

class Unit(BaseModel):
    """Represents a single unit in the syllabus, containing multiple topics."""
    title: str = Field(description="The title of the unit")
    topics: List[str] = Field(description="A list of topics that belong to the unit")

class Syllabus(BaseModel):
    """Represents the entire syllabus, structured into multiple units."""
    units: List[Unit] = Field(description="A list of units in the syllabus, each containing topics")

# Function to process the syllabus using structured output from the LLM
def syllabus_llm(syllabus_text: str):
    """
    This function will invoke the model to process and return structured syllabus data.
    """
    structured_llm = llm.with_structured_output(Syllabus)

    # Updated prompt for the LLM
    prompt = f"""Analyze and structure the following syllabus content into a clear, hierarchical format:
1. Identify the main subject areas or units. These should be broad categories that encompass multiple related topics.
2. Within each unit, list specific topics that are part of that unit.
Rules:
- Each unit should represent a distinct area of study or a major theme.
- Topics should be specific subjects or concepts within the unit.
- If a line contains a main subject followed by subtopics (e.g., "CSS: introduction, rules"), treat the main subject as a unit and create separate, detailed topics for each subtopic.
- Ensure each topic is clearly defined, specific, and easily retrievable using similarity search algorithms.
- Break down broad topics into more specific, searchable phrases. For example, instead of just "CSS", use "Introduction to CSS", "CSS Ruleset Structure", "Inline CSS Techniques", etc.
- Make sure each topic title is descriptive and self-contained, avoiding vague terms like "introduction" or "basics" without context.
- Aim for topic titles that are 2-5 words long, balancing specificity with conciseness.
Here's the syllabus content to structure:
{syllabus_text}
Please format the output as a Syllabus with Units and Topics, adhering to the structure defined by the Pydantic models. Ensure that the topics are detailed enough to be easily retrieved from a vector database using similarity search algorithms."""

    # Invoke the model with structured output
    response = structured_llm.invoke(prompt)

    return response