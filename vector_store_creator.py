from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def create_vector_store(chunks, api_key):
    """Creates embeddings and a FAISS vector store using GoogleGenerativeAIEmbeddings."""
    if not chunks:
        print("No chunks provided to create vector store.")
        return None
    try:
        print("Creating embeddings and vector store...")

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", # Specify the embedding model
            google_api_key=api_key
        )

        print("Embedding documents...")
        vector_store = FAISS.from_documents(chunks, embeddings)
        print("Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store or embeddings: {e}")
        if "API key not valid" in str(e):
             print("Please check if your GOOGLE_API_KEY is correct and has permissions for the embedding model.")
        elif "403" in str(e) and "permission" in str(e).lower():
             print("Error 403: Permission denied. Ensure the API key is enabled for the 'Generative Language API' in your Google Cloud project.")
        elif "resource has been exhausted" in str(e) or "429" in str(e):
             print("You might have hit API rate limits (Error 429). Please wait and try again later or check your quota.")
        elif "Model not found" in str(e):
             print("Error: Embedding model 'models/embedding-001' not found or unavailable with your API key.")
        else:
            # Print detailed traceback for unexpected errors
            import traceback
            traceback.print_exc()
        return None