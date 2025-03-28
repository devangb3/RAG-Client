import os
import argparse
import warnings
from dotenv import load_dotenv
from vector_store_creator import create_vector_store
from file_processor import load_and_split_document
from answer_generator import process_query
warnings.filterwarnings("ignore", category=UserWarning, message=".*Could not import google-auth.*")

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

def main():
    parser = argparse.ArgumentParser(description="RAG CLI tool using Gemini API")
    parser.add_argument("filepath", help="Path to the text document to process.")
    args = parser.parse_args()

    doc_chunks = load_and_split_document(args.filepath)
    if not doc_chunks:
        return
    vector_store = create_vector_store(doc_chunks, api_key)
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

            answer = process_query(query, vector_store)
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