import os
from dotenv import load_dotenv
import google.generativeai as genai
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

genai.configure(api_key=api_key)

def retrieve_context(query, vector_store, k):
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

    prompt = f"""Based *only* on the following context extracted from the document, please answer the question. If the context doesn't contain the answer,
    state that the information is not available in the provided text. Do not use any prior knowledge.
    Context:
    {context_text}

    Question: {query}

    Answer:"""

    try:
        model = genai.GenerativeModel('gemini-2.5-pro-exp-03-25')
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
             return f"Error: Generation model specified ('gemini-2.5-pro-exp-03-25' or 'gemini-pro') not found or unavailable."
        return f"An error occurred during generation: {e}"

def process_query(query, vector_store, relevant_chunks):
    context = retrieve_context(query, vector_store, relevant_chunks)
    answer = generate_answer(query, context)

    return answer