#D:\text_to_sql_bot\check.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

# Create the model client
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    api_key=api_key
)

# Invoke the model
response = llm.invoke("What is the captial of india.")
print(response.content)
