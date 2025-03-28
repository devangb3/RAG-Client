# RAG CLI Tool

A command-line interface tool that implements Retrieval-Augmented Generation (RAG) using Google's Gemini API. This tool allows you to ask questions about the content of text and PDF documents.

## Features

- Process single documents or entire directories
- Supports both `.txt` and `.pdf` files
- Uses Google's Gemini API for text generation
- Implements RAG pattern using FAISS vector store
- Interactive Q&A interface

## Prerequisites

- Python 3.8 or higher
- Google API key with access to Gemini API

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install langchain langchain-google-genai python-dotenv faiss-cpu pypdf
   ```
4. Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

## Usage

### Processing a Single File

```bash
python rag_cli.py path/to/your/document.txt
```

### Processing a Directory

```bash
python rag_cli.py path/to/your/directory
```

### Interactive Mode

After processing the document(s), you can:
- Type your questions about the document content
- Get AI-generated answers based on the document context
- Type 'quit' or 'exit' to end the session

## How It Works

1. **Document Processing**: 
   - Loads documents using LangChain's document loaders
   - Splits documents into manageable chunks

2. **Vector Store Creation**:
   - Creates embeddings using Google's embedding model
   - Stores embeddings in a FAISS vector store

3. **Query Processing**:
   - Retrieves relevant document chunks for each query
   - Generates contextual answers using Gemini API

## Error Handling

The tool includes comprehensive error handling for:
- Invalid API keys
- Missing files or directories
- Unsupported file types
- Password-protected PDFs
- API rate limits

## Limitations

- Only supports .txt and .pdf files
- PDF processing ignores images
- Requires active internet connection
- Subject to Google API rate limits
