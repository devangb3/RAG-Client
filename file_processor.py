import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, PythonLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PythonLoader

def process_directory(directory_path):
    """
    Process all supported files (.txt, .pdf, .py) in a directory
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Path provided is not a valid directory: {directory_path}")
        return None
    if not os.path.exists(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return None

    print(f"\n--- Starting Directory Processing using DirectoryLoader: {directory_path} ---")
    all_loaded_documents = []
    total_files_processed = 0

    try:
        print("Loading .txt files...")
        loader_txt = DirectoryLoader(
            directory_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'},
            recursive=True,
            show_progress=True,
            use_multithreading=True
        )
        docs_txt = loader_txt.load()
        all_loaded_documents.extend(docs_txt)
        total_files_processed += len(docs_txt)
        print(f"-> Loaded {len(docs_txt)} .txt documents.")
    except Exception as e:
        print(f"Warning: Error loading .txt files: {e}")

    try:
        print("Loading .pdf files...")
        loader_pdf = DirectoryLoader(
            directory_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            recursive=True,
            show_progress=True,
            use_multithreading=True
        )
        docs_pdf = loader_pdf.load()
        all_loaded_documents.extend(docs_pdf)
        print(f"-> Loaded {len(docs_pdf)} .pdf document pages/sections.")
    except ImportError:
         print("Warning: 'pypdf' not installed. Skipping PDF loading. Run 'pip install pypdf'.")
    except Exception as e:
        print(f"Warning: Error loading .pdf files: {e}. Check file integrity and passwords.")

    try:
        print("Loading .py files (as text)...")
        loader_py = DirectoryLoader(
            directory_path,
            glob="**/*.py",       
            loader_cls=PythonLoader,
            recursive=True,
            show_progress=True,
            use_multithreading=True
        )
        docs_py = loader_py.load()
        all_loaded_documents.extend(docs_py)
        total_files_processed += len(docs_py)
        print(f"-> Loaded {len(docs_py)} .py documents.")
    except Exception as e:
        print(f"Warning: Error loading .py files: {e}")

    print(f"\n--- Directory Loading Complete ---")
    print(f"Total documents/pages loaded across all types: {len(all_loaded_documents)}")

    if not all_loaded_documents:
        print("No documents were successfully loaded from the directory.")
        return None

    all_chunks = split_document(all_loaded_documents)

    if not all_chunks:
         print("No chunks were created after splitting. Check loaded content.")
         return None

    print(f"Total chunks created from all files: {len(all_chunks)}")
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

    if not documents:
        return []
    
    print("Splitting document into smaller chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
         print("Warning: Document splitting resulted in zero chunks. Check document content and splitter settings.")
         return []
    print(f"Split into {len(chunks)} chunks.")
    return chunks