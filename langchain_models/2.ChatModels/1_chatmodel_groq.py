# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# load_dotenv()

# llm = ChatGroq(model="llama-3.3-70b-versatile",
#     temperature=0, max_tokens=100)
# result =llm.invoke("what is India's GDP?")
# print(result.content)


from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    max_tokens=500
)

result = llm.invoke("What is India's GDP?")

print(result.content)
