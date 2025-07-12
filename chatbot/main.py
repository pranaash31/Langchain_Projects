from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Set Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

#  Optional: Set LangSmith (LangChain Tracing)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries."),
        ("user", "Question: {question}")

    ]
)

#Streamlit Framework
st.title("LangChain Demo with Groq API key")
input_text = st.text_input("Search the topic you want")

#GroqAPIkey
llm = ChatGroq(
    model_name="llama3-70b-8192",  # or "mixtral-8x7b-32768"
    groq_api_key=os.getenv("GROQ_API_KEY")
)
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))


