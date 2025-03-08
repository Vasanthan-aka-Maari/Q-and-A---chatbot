import streamlit as st
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

# LangSmith Tracking
os.environ['LANGSMITH_API_KEY'] = langsmith_api_key
os.environ['LANGSMITH_TRACING_V2'] = "true"
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "Q&A Chatbot"

# Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an intelligent system that assists with all things technical. Kindly respond to the user's questions."),
        ("user", "Question: {question}")
    ]
)

parser = StrOutputParser()

def generate_response(question, api_key, model):
    llm = ChatGroq(groq_api_key=api_key, model_name=model)
    chain = prompt | llm | parser
    response = chain.invoke({"question": question})
    return response

# Streamlit setup
st.title("Enhanced Q&A Chatbot with Groq")

# Sidebar for settings
st.sidebar.title("Settings")
api_key=st.sidebar.text_input("Enter your Groq API key: ", type="password")
model=st.sidebar.selectbox("Select a model:", ["llama-3.3-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"])

# Main interface
st.write("Shoot your question")
question = st.text_input("You:")

if question:
    response = generate_response(question, api_key, model)
    st.write(response)