from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Location of our vector database (chromaDB)
persistent_directory = "db/chrome_db"

# Ollama local embedding model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# Creating Chroma db instance
db = Chroma(
    persist_directory = persistent_directory,
    embedding_function = embedding_model,
    collection_metadata = {"hnsw:space": "cosine"}
)


# User query to do retrieval from private database
query = "When and for how much money microsoft paid to buy github ??"

# defining retriever for top k results
retriever = db.as_retriever(search_kwargs={"k": 5})

# using retriever to get top k related chunks from vector DB
relevant_docs = retriever.invoke(query)

# printing the result
print(f"User Query: {query}")
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")


