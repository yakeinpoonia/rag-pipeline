import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv


# Function to laod files from private_data folder
def load_documents(docs_path="private_data"):
    print(f"Loading documents form {docs_path}.....")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        message = f"The directory {docs_path} does not exist."
        error = FileNotFoundError(message)
        raise error

    # Load all .txt files from the docs_path directory
    loader = DirectoryLoader(
        path = docs_path,
        glob = "*.txt",
        loader_cls = TextLoader,
        show_progress = True
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No .txt file found in {docs_path}.")

    for i, doc in enumerate(documents):
        print(f"\nDocument {i+1}: ")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length: {len(doc.page_content)} characters")

    return documents

# Fucntion to split large .txt files into smaller chunk
def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("\nSplitting documents into chunks....\n")

    # CharacterTextsplitter is a class
    text_splitter = CharacterTextSplitter(
        separator = ".",
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    # split_documents is a method inherited from TextSplitter
    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 90)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

# Function to create embeddings of chunks and storing them in Vector DB
def create_vector_store(chunks, persist_directory="db/chrome_db"):
    print("\nCreating embeddings and storing in ChromeDB....\n")

    # Using ollama local model for embeddings
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Create ChromeDB vector store
    print("\n--- Creating vectore store ---\n")
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata = {"hnsw:space": "cosine"} # By default chroma uses L2 (Euclidean) distance
    )
    print("\n--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    # Loading documents from private database
    documents = load_documents(docs_path="private_data")

    # Chunking large files
    chunks = split_documents(documents)

    # Embedding and Storing in Vector DB
    vectorstore = create_vector_store(chunks)



if __name__ == "__main__":
    main()
