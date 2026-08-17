from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", dimensions=32
)

documents = [
    "I am a boy",
    "I play football",
    "I work as an ai engineer"
]

result = embeddings.embed_documents(documents)

print(str(result))
