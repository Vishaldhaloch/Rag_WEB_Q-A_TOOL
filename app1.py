import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import re
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from urllib.parse import urlparse

# Page title and description
st.set_page_config(page_title="Web Content Q&A Tool", layout="wide")
st.title("Web Content Q&A Tool")
st.markdown("""
This application allows you to input URLs and ask questions that will be answered using only the content from those webpages.
The app uses a Retrieval-Augmented Generation (RAG) architecture to provide accurate answers based solely on the provided content.
""")

# Initialize session state for storing scraped content and vector store
if 'content_dict' not in st.session_state:
    st.session_state.content_dict = {}
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'urls_processed' not in st.session_state:
    st.session_state.urls_processed = []

# Function to clean HTML content
def clean_text(text):
    # Remove extra whitespace (but preserve paragraph breaks)
    text = re.sub(r'\n+', '\n', text)  # Normalize line breaks
    text = re.sub(r' +', ' ', text)    # Normalize spaces
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Keep some special characters important for meaning but remove others
    text = re.sub(r'[^\w\s.?!,;:()\-\'\"\n]', '', text)
    
    return text.strip()

# Function to extract main content from HTML
def extract_main_content(soup):
    # Priority containers that typically contain main content
    main_tags = ['article', 'main', '.content', '.post', '.entry', '#content', '.post-content']
    
    # Try to find the main content container
    main_content = None
    for tag in main_tags:
        if tag.startswith('.'):
            main_content = soup.select(tag)
        elif tag.startswith('#'):
            main_content = soup.select(tag)
        else:
            main_content = soup.find_all(tag)
        
        if main_content and len(main_content) > 0:
            break
    
    # If we found a main content container, use it
    if main_content and len(main_content) > 0:
        # Get the largest content block if multiple were found
        if len(main_content) > 1:
            main_content = max(main_content, key=lambda x: len(x.get_text()))
        else:
            main_content = main_content[0]
        
        # Remove unwanted elements from the main content
        for unwanted in main_content.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            unwanted.decompose()
            
        return main_content.get_text()
    
    # If no main content container was found, extract from the body with filtering
    else:
        # Remove unwanted elements from the entire body
        for unwanted in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
            unwanted.decompose()
        
        # Extract all paragraphs and headings as these usually contain the main content
        content_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div.content', 'article'])
        
        # Filter out elements with very little text
        content_elements = [el for el in content_elements if len(el.get_text().strip()) > 40]
        
        if content_elements:
            return '\n\n'.join([el.get_text().strip() for el in content_elements])
        else:
            # Fallback: just get all text from the body
            body = soup.find('body')
            return body.get_text() if body else soup.get_text()

# Function to scrape and process a URL
def scrape_url(url):
    try:
        # Check if URL is valid
        parsed_url = urlparse(url)
        if not all([parsed_url.scheme, parsed_url.netloc]):
            return f"Invalid URL format: {url}"
        
        # Add http:// prefix if missing
        if not url.startswith('http'):
            url = 'https://' + url

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return f"Failed to retrieve content from {url}. Status code: {response.status_code}"
        
        # Parse HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content
        content = extract_main_content(soup)
        
        # Clean the text
        cleaned_text = clean_text(content)
        
        # Store the content
        st.session_state.content_dict[url] = cleaned_text
        
        # Log some info for debugging
        content_preview = cleaned_text[:100] + "..." if len(cleaned_text) > 100 else cleaned_text
        return f"Successfully scraped {url} ({len(cleaned_text)} characters)\nPreview: {content_preview}"
    
    except Exception as e:
        return f"Error scraping {url}: {str(e)}"

# Function to create vector store from scraped content
def create_vectorstore():
    if not st.session_state.content_dict:
        return "No content to process. Please scrape URLs first."
    
    try:
        # Create a mapping from chunks to source URLs
        chunks_with_sources = []
        metadata_list = []
        
        # Process each URL's content separately to track source
        for url, content in st.session_state.content_dict.items():
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            url_chunks = text_splitter.split_text(content)
            
            # Create metadata for each chunk
            for chunk in url_chunks:
                chunks_with_sources.append(chunk)
                metadata_list.append({"source": url})
        
        if not chunks_with_sources:
            return "No content chunks created. The scraped content may be too short."
        
        # Create embeddings and vector store with metadata
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_texts(
            texts=chunks_with_sources,
            embedding=embeddings,
            metadatas=metadata_list
        )
        st.session_state.urls_processed = list(st.session_state.content_dict.keys())
        
        return f"Successfully processed {len(chunks_with_sources)} content chunks from {len(st.session_state.content_dict)} URLs."
    
    except Exception as e:
        return f"Error creating vector store: {str(e)}"

# Function to generate answer from RAG pipeline
def generate_answer(question):
    if not st.session_state.vectorstore:
        return "No vector store available. Please process URLs first."
    
    try:
        # Get API key from environment or user input
        api_key = os.environ.get("GROQ_API_KEY") or st.session_state.get("api_key", "")
        if not api_key:
            return "Groq API key not provided. Please enter it in the sidebar."
        
        # Create LLM
        llm = ChatGroq(
            api_key=api_key,
            model_name="llama3-8b-8192",  # Using Llama 3 model via Groq
        )
        
        # Create retriever with metadata
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # Get retrieved documents to display sources
        retrieved_docs = retriever.get_relevant_documents(question)
        
        # Track which URLs were used in the response
        used_sources = set()
        for doc in retrieved_docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                used_sources.add(doc.metadata['source'])
        
        # Store for displaying to the user
        st.session_state.used_sources = list(used_sources)
        
        # Format the context from retrieved documents
        context_texts = [doc.page_content for doc in retrieved_docs]
        formatted_context = "\n\n".join(context_texts)
        
        # Create prompt template with improved instructions
        template = """You are an AI assistant that answers questions based SOLELY on the provided context.
        If the answer cannot be determined from the context, say "I cannot answer this question based on the provided content."
        DO NOT use any information outside of the context provided below.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer the question thoroughly and accurately using only information from the context. If the context doesn't contain relevant information, acknowledge the limitations.
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create RAG pipeline
        rag_chain = (
            {"context": lambda x: formatted_context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Generate answer
        answer = rag_chain.invoke(question)
        
        return answer
    
    except Exception as e:
        return f"Error generating answer: {str(e)}"

# Sidebar for API key
with st.sidebar:
    st.header("API Configuration")
    api_key = st.text_input("Enter Groq API Key", type="password")
    if api_key:
        st.session_state["api_key"] = api_key
    
    st.divider()
    st.header("Current Status")
    st.write(f"URLs processed: {len(st.session_state.urls_processed)}")
    if st.session_state.urls_processed:
        st.write("Processed URLs:")
        for url in st.session_state.urls_processed:
            st.write(f"- {url}")
    
    if st.button("Clear All Data"):
        st.session_state.content_dict = {}
        st.session_state.vectorstore = None
        st.session_state.urls_processed = []
        st.success("All data cleared!")

# URL input section
st.header("Step 1: Input URLs")
url_input = st.text_area("Enter URLs (one per line)", height=100)
scrape_button = st.button("Scrape URLs")

if scrape_button and url_input:
    urls = [url.strip() for url in url_input.split('\n') if url.strip()]
    if urls:
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        
        for i, url in enumerate(urls):
            status_placeholder.info(f"Scraping {url}...")
            result = scrape_url(url)
            st.write(result)
            progress_bar.progress((i + 1) / len(urls))
        
        status_placeholder.success("URL scraping completed!")
        
        # Process content into vector store
        with st.spinner("Creating vector store..."):
            result = create_vectorstore()
            st.write(result)
    else:
        st.warning("Please enter at least one valid URL.")

# Question answering section
st.header("Step 2: Ask Questions")
question = st.text_input("Enter your question about the scraped content")
answer_button = st.button("Get Answer")

if answer_button and question:
    if not st.session_state.vectorstore:
        st.warning("Please scrape and process URLs before asking questions.")
    else:
        with st.spinner("Generating answer..."):
            answer = generate_answer(question)
            
            st.subheader("Answer")
            st.write(answer)
            
            st.subheader("Sources")
            if hasattr(st.session_state, 'used_sources') and st.session_state.used_sources:
                st.write("Answer derived from these specific sources:")
                for url in st.session_state.used_sources:
                    st.write(f"- {url}")
            else:
                st.write("Based on content from:")
                for url in st.session_state.urls_processed:
                    st.write(f"- {url}")
                    
            # Add content preview option
            if st.checkbox("Show content preview"):
                st.subheader("Content Preview")
                for url, content in st.session_state.content_dict.items():
                    with st.expander(f"Content from {url}"):
                        st.text_area("Scraped content", value=content[:1000] + "..." if len(content) > 1000 else content, height=200)