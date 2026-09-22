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




def main():
    documents = load_documents(docs_path="private_data")


if __name__ == "__main__":
    main()
