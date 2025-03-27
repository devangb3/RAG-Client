import os
import argparse
import warnings
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not import google-auth.*")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

genai.configure(api_key=api_key)


def load_and_split_document(filepath):
    """Loads a text document and splits it into chunks."""
    print(f"Loading document from: {filepath}")
    try:
        loader = TextLoader(filepath, encoding='utf-8')
        documents = loader.load()
        if not documents:
            print("Error: No document content loaded.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split document into {len(chunks)} chunks.")
        return chunks
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading or splitting document: {e}")
        return None

def create_vector_store(chunks):
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



def retrieve_context(query, vector_store, k=3):
    """Retrieves relevant context chunks from the vector store."""
    if vector_store is None:
        print("Vector store is not available for retrieval.")
        return []
    print(f"\nRetrieving context for query: '{query}'")
    try:
        # Perform similarity search
        results = vector_store.similarity_search(query, k=k)
        print(f"Retrieved {len(results)} relevant chunks.")
        return results
    except Exception as e:
        print(f"Error during context retrieval: {e}")
        return []

def generate_answer(query, context_chunks):
    """Generates an answer using Gemini based on the query and context."""
    if not context_chunks:
        print("No context provided. Generating answer based on query alone (may be less accurate).")
        context_text = "No specific context provided."
    else:
        context_text = "\n\n".join([chunk.page_content for chunk in context_chunks])

    prompt = f"""Based on the following context, please answer the question. If the context doesn't contain the answer, say you don't know based on the provided text.

Context:
{context_text}

Question: {query}

Answer:"""

    print("Generating answer using Gemini...")
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt)

        if not response.parts:
             if response.prompt_feedback.block_reason:
                  print(f"Warning: Response blocked due to {response.prompt_feedback.block_reason}")
                  return "Response was blocked due to safety settings."
             else:
                  print("Warning: Received an empty response from the model.")
                  return "Model returned an empty response."

        return response.text.strip()

    except Exception as e:
        print(f"Error generating answer with Gemini: {e}")
        if "API key not valid" in str(e):
             return "Error: Invalid API Key for generation model."
        elif "resource has been exhausted" in str(e):
             return "Error: API rate limit exceeded for generation."
        elif "Model not found" in str(e):
             return f"Error: Generation model specified ('gemini-1.5-flash-latest' or 'gemini-pro') not found or unavailable."
        return f"An error occurred during generation: {e}"



def main():
    parser = argparse.ArgumentParser(description="RAG CLI tool using Gemini API")
    parser.add_argument("filepath", help="Path to the text document to process.")
    args = parser.parse_args()

    doc_chunks = load_and_split_document(args.filepath)
    if not doc_chunks:
        return
    vector_store = create_vector_store(doc_chunks)
    if not vector_store:
        return

    print("\n--- RAG CLI Ready ---")
    print("Ask questions about the document. Type 'quit' or 'exit' to stop.")

    while True:
        try:
            query = input("\nYour Question: ")
            if query.lower() in ['quit', 'exit']:
                break
            if not query:
                continue

            context = retrieve_context(query, vector_store)

            answer = generate_answer(query, context)
            print(f"\nAnswer:\n{answer}")

        except EOFError:
             break
        except KeyboardInterrupt:
             print("\nExiting...")
             break
        except Exception as e:
             print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    main()