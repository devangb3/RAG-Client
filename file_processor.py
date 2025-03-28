import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def process_directory(directory_path):
    """Process all supported files in a directory and its subdirectories."""
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return None

    all_chunks = []
    supported_extensions = {'.txt', '.pdf'}
    
    for root, _, files in os.walk(directory_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in supported_extensions:
                filepath = os.path.join(root, file)
                chunks = load_and_split_document(filepath)
                if chunks:
                    all_chunks.extend(chunks)
    
    print(f"Processed all files in directory. Total chunks: {len(all_chunks)}")
    return all_chunks

def load_and_split_document(filepath):
    """Loads a text document and splits it into chunks."""

    print(f"Loading document from: {filepath}")
    file_extension = os.path.splitext(filepath)[1].lower()

    try:
        if file_extension == ".txt":
            loader = TextLoader(filepath, encoding='utf-8')
            print("Using TextLoader for .txt file.")
        elif file_extension == ".pdf":
            loader = PyPDFLoader(filepath)
            print("Using PyPDFLoader for .pdf file. Images will be ignored.")
        else:
            print(f"Error: Unsupported file type '{file_extension}'. Only .txt and .pdf are supported.")
            return None

        documents = loader.load()
        if not documents:
            print("Error: No document content loaded. The file might be empty, corrupted, or password-protected (for PDFs).")
            return None
        print(f"Successfully loaded {len(documents)} initial document pages/sections.")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading document: {e}")
        if "password" in str(e).lower():
             print("Hint: The PDF might be password-protected.")
        return None
    chunks = split_document(documents)
    return chunks
    

def split_document(documents):

    print("Splitting document into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
         print("Warning: Document splitting resulted in zero chunks. Check document content and splitter settings.")
         return None # Or handle as appropriate

    print(f"Split document into {len(chunks)} chunks.")
    return chunks
