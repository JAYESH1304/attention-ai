import streamlit as st
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from neo4j import GraphDatabase
from transformers import T5ForConditionalGeneration, T5Tokenizer
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer, util
import os
import faiss
from langchain.schema import Document
import pandas as pd
import groq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
import re
import fitz  # PyMuPDF for PDF text extraction
import os
from typing import List, Dict
import arxiv
from langchain_groq import ChatGroq
from groq import Groq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

groq_api_key = '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'

# Neo4j database connection function
def connect_to_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    return driver

def store_papers_in_neo4j(driver, papers):
    with driver.session() as session:
        for paper in papers:
            session.run(
                """
                MERGE (p:Paper {title: $title, summary: $summary, link: $link, year: $year})
                """,
                title=paper["title"],
                summary=paper["summary"],
                link=paper["link"],
                year=paper["year"]
            )

def fetch_arxiv_papers(topic, start_year):
    base_url = 'http://export.arxiv.org/api/query?search_query=all:'
    end_year = start_year + 5
    url = f"{base_url}{topic}&start=0&max_results=50&date_range={start_year}0101-{end_year}1231"
    response = requests.get(url)
    
    if response.status_code != 200:
        return f"Error fetching data from ArXiv API. Status Code: {response.status_code}"

    root = ET.fromstring(response.content)
    papers = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text
        link = entry.find('{http://www.w3.org/2005/Atom}id').text
        published = entry.find('{http://www.w3.org/2005/Atom}published').text
        published_year = int(published.split('-')[0])

        if published_year >= start_year and published_year <= end_year:
            papers.append({
                "title": title, 
                "summary": summary, 
                "link": link, 
                "year": published_year
            })
    
    return papers

def download_arxiv_paper(link):
    paper_id = re.search(r'arxiv.org/abs/([0-9.]+)', link).group(1)
    pdf_link = f"https://arxiv.org/pdf/{paper_id}.pdf"
    
    # Define the local directory to save PDFs
    download_dir = "downloaded_papers"
    os.makedirs(download_dir, exist_ok=True)  # Create the directory if it doesn't exist
    pdf_path = os.path.join(download_dir, f"{paper_id}.pdf")

    response = requests.get(pdf_link)
    if response.status_code == 200:
        with open(pdf_path, 'wb') as f:
            f.write(response.content)
        return pdf_path
    else:
        return None

def extract_text_from_pdf(pdf_path):
    # Extract text from the PDF file
    full_text = ""
    with fitz.open(pdf_path) as pdf_document:
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            full_text += page.get_text()
    return full_text




# Load the Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def query_relevant_papers(driver, user_query):
    user_query_embedding = sbert_model.encode(user_query, convert_to_tensor=True)
    
    with driver.session() as session:
        result = session.run("MATCH (p:Paper) RETURN p.title AS title, p.summary AS summary, p.link AS link")
        
        papers = []
        for record in result:
            title = record["title"]
            summary = record["summary"]
            link = record["link"]
            paper_embedding = sbert_model.encode(summary, convert_to_tensor=True)
            similarity_score = util.pytorch_cos_sim(user_query_embedding, paper_embedding).item()
            
            papers.append({"title": title, "summary": summary, "link": link, "similarity": similarity_score})
    
    papers = sorted(papers, key=lambda x: x["similarity"], reverse=True)
    return papers if papers else None


from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
    """
    You are a research paper assistant.
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    Question:{input}

    """
)

def create_vector_embedding(input_text: str):
    if "vectors" not in st.session_state:
        # Initialize embeddings and text splitter
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Prepare and split the input text
        docs = [Document(page_content=input_text)]
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)

        # Create vector embeddings and store them
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
    return st.session_state.vectors



def generate_future_directions(papers: List[Dict]) -> str:
    """Generate future research directions based on paper summaries."""
    summaries = "\n".join([f"Title: {p['title']}\nSummary: {p['summary']}\n" for p in papers[:1]])
    
    prompt = f"""Based on these recent research papers:
    {summaries}
    
    Please analyze the current trends and suggest 3-5 promising future research directions.
    For each direction, explain:
    1. The motivation
    2. Potential impact
    3. Technical challenges to overcome"""
    try:
        # document_chain = create_stuff_documents_chain(llm, prompt)
        response = llm.invoke(prompt)  # Assuming it works directly for a generation process
        return response
       
    except Exception as e:
        print(f"Error generating future directions: {e}")
        return "Error generating future research directions."


def main():
    st.title("Academic Research Paper Assistant Application")
    
    neo4j_uri = "neo4j+s://08604d40.databases.neo4j.io"
    neo4j_user = "neo4j"
    neo4j_password = "TKs7thScxvjWy62MdpKqNkTQw97-gD5V6ENJt7n4DQo"

    first_query = st.text_input("Enter your first query (e.g., 'show me all the papers related to machine learning in the last 5 year'):")
    if first_query:
        topic = re.search(r'related to (.*) in the last', first_query).group(1)
        current_year = datetime.now().year
        start_year = current_year - 5
        
        if st.button("Fetch and Store Papers"):
            with st.spinner("Fetching papers..."):
                papers = fetch_arxiv_papers(topic, start_year)

                if isinstance(papers, str):
                    st.error(papers)
                elif not papers:
                    st.info("No papers found for the given topic and year range.")
                else:
                    st.header("Fetched Papers")

                    # Create a DataFrame from the papers list
                    papers_df = pd.DataFrame(papers)
                    papers_df['year'] = papers_df['year'].astype(int)

                    # Display the DataFrame in a Streamlit dataframe
                    st.dataframe(papers_df[['title', 'year']])

                    with st.spinner("Connecting to Neo4j and storing data..."):
                        driver = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)
                        try:
                            store_papers_in_neo4j(driver, papers)
                            st.success("Papers successfully stored in the Neo4j database.")
                        except Exception as e:
                            st.error(f"An error occurred while storing data in Neo4j: {e}")
                        finally:
                            driver.close()
    else:
        st.warning("Please enter your first query.")


    st.header("User Query and Generate Answer")
    user_query = st.text_input("Enter your query:")
    
    most_relevant_paper = None

    # if st.button("Generate Answer"):
    #     if user_query:
    #         with st.spinner("Querying the database and generating answer..."):
    #             driver = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)
    #             try:
    #                 most_relevant_paper = query_relevant_papers(driver, user_query)
    #                 most_relevant_paper = most_relevant_paper[0]
    #                 if most_relevant_paper:
    #                     st.write(f"**Most Relevant Paper for above query**: {most_relevant_paper['title']}")
    #                     st.write(f"**Summary**: {most_relevant_paper['summary']}")
    #                     st.write(f"**Link**: [View Paper]({most_relevant_paper['link']})")
                        
    #                     pdf_path = download_arxiv_paper(most_relevant_paper["link"])
    #                     if pdf_path:
    #                         st.write("**Downloaded Paper PDF**:", pdf_path)
    #                         context = extract_text_from_pdf(pdf_path)
    #                         create_vector_embedding(context)
                            
    #                         document_chain=create_stuff_documents_chain(llm,prompt)
    #                         retriever=st.session_state.vectors.as_retriever()
    #                         retrieval_chain=create_retrieval_chain(retriever,document_chain)
    #                         response=retrieval_chain.invoke({'input':user_query})
    #                         st.write("**Answer to Your Query:**")
    #                         st.write(response['answer'])
                    
    #                 else:
    #                     st.info("No relevant papers found for the query.")
    #             except Exception as e:
    #                 st.error(f"An error occurred while querying data: {e}")
    #             finally:
    #                 driver.close()
                    
    if st.button("Generate Future Directions"):
        if user_query:
            with st.spinner("Generating future research directions..."):
                driver = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)
                try:
                    most_relevant_papers = query_relevant_papers(driver, user_query)
                    if most_relevant_papers:
                        future_directions = generate_future_directions(most_relevant_papers[0])
                        st.write("**Future Research Directions:**")
                        st.write(future_directions)
                    else:
                        st.info("No relevant papers found for the query.")
                except Exception as e:
                    st.error(f"An error occurred while generating future directions: {e}")
                finally:
                    driver.close()

if __name__ == "__main__":
    main()