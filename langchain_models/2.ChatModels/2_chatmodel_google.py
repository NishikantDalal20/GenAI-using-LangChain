from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    max_tokens=100
)

result = model.invoke("Who is Narendra Modi?")

print(result.content)