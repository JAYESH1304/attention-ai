
# Academic Research Paper Assistant Application

This Streamlit-based application assists users in querying academic research papers, extracting and storing data from ArXiv, and generating answers and future research directions using an integrated NLP pipeline.

[Watch the project demo on YouTube](https://www.youtube.com/watch?v=wMX306g5tf0)

## Features

- **Fetch and Store ArXiv Papers**: Retrieve academic papers from ArXiv based on user-defined topics and date ranges.
- **Neo4j Integration**: Store and query academic paper data using a connected Neo4j database.
- **PDF Download and Text Extraction**: Download and extract text from ArXiv papers' PDFs using PyMuPDF.
- **NLP-Based Querying**: Use Sentence-BERT for querying and finding the most relevant papers from the Neo4j database.
- **Vector Embedding Creation**: Generate vector embeddings for text-based content using FAISS.
- **Future Research Direction Suggestions**: Generate future research directions based on paper summaries using Groq LLM.

## Installation

### Prerequisites

Ensure you have the following installed:

- [Python 3.8+](https://www.python.org/)
- [Neo4j](https://neo4j.com/download/)
- [Streamlit](https://streamlit.io/)
- [Transformers](https://huggingface.co/transformers/)
- [LangChain](https://langchain.com/)
- [FAISS](https://faiss.ai/)
- [Sentence-Transformers](https://www.sbert.net/)
- [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd <project-directory>
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application using Streamlit:
   ```bash
   streamlit run app.py
   ```
2. Enter your query to fetch academic papers, store them, and generate responses using NLP pipelines.

### Environment Variables

Make sure to set the required environment variables before running the app:

- `groq_api_key`: Your API key for using the Groq LLM.

## Key Components

- **Neo4j Connection**: The app connects to a Neo4j database for storing and querying paper data.
- **ArXiv API Integration**: Fetches papers from ArXiv based on user-defined queries.
- **PDF Download and Text Extraction**: Downloads PDFs and extracts text using PyMuPDF.
- **NLP Query Processing**: Leverages Sentence-BERT for querying and Groq LLM for generating responses and future research directions.
- **Vector Embeddings**: Utilizes FAISS for vector storage and retrieval of document embeddings.

## Project Structure

```plaintext
.
├── app.py                   # Main application script
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

Thank you for checking out my project! Feel free to explore, contribute, or reach out if you have any questions.

