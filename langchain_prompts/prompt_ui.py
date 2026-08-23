from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

st.header('Reasearch Tool')
user_input = st.text_input('Enter your prompt')

llm = HuggingFaceEndpoint(
    #repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")

print(result.content)