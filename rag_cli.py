import os
import argparse
import warnings
from dotenv import load_dotenv
from vector_store_creator import create_vector_store
from file_processor import load_and_split_document, process_directory
from answer_generator import process_query

warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not import google-auth.*")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

def main():
    parser = argparse.ArgumentParser(description="RAG CLI tool using Gemini API")
    parser.add_argument("filepath", help="Path to a file or directory to process (supports .txt and .pdf files)")
    args = parser.parse_args()

    if os.path.isdir(args.filepath):
        doc_chunks = process_directory(args.filepath)
    else:
        doc_chunks = load_and_split_document(args.filepath)

    if not doc_chunks:
        return

    vector_store = create_vector_store(doc_chunks, api_key)
    if not vector_store:
        return

    print("\n--- RAG CLI Ready ---")
    print("Ask questions about the document(s). Type 'quit' or 'exit' to stop.")

    while True:
        try:
            query = input("\nYour Question: ")
            if query.lower() in ['quit', 'exit']:
                break
            if not query:
                continue
            relevant_chunks = input("\nNumber of relevant chunks to retireve from vector store : ")
            if not relevant_chunks:
                relevant_chunks = 5
            answer = process_query(query, vector_store, int(relevant_chunks))
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