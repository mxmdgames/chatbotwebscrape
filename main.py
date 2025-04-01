import streamlit as st
import ollama
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from urllib.parse import urlparse
import re

# Configure Streamlit page
st.set_page_config(
    page_title="Ollama Search Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Available models
AVAILABLE_MODELS = [
    'llama3.1:latest',

]

# Session state initialization
if 'web_content' not in st.session_state:
    st.session_state.web_content = ""
if 'response' not in st.session_state:
    st.session_state.response = ""
if 'processing' not in st.session_state:
    st.session_state.processing = False

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def scrape_page(url, max_length):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        for element in soup(['script', 'style', 'nav', 'footer', 'iframe', 'noscript']):
            element.decompose()

        text = soup.get_text(separator='\n', strip=True)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text[:max_length]
    except Exception as e:
        st.error(f"Error scraping {url}: {e}")
        return None

def get_web_context(prompt, search_count, max_length):
    try:
        with DDGS() as ddgs:
            results = []
            for result in ddgs.text(prompt):
                if len(results) >= search_count:
                    break
                results.append(result)

        context = ""
        for result in results:
            if not is_valid_url(result['href']):
                continue

            content = scrape_page(result['href'], max_length)
            if content:
                context += f"\n\n--- PAGE: {result['title']} ---\n{content}\n"

        return context if context else "No relevant web content found."
    except Exception as e:
        return f"Web search error: {e}"

def generate_response(prompt, model, use_internet, search_count, max_length):
    try:
        if use_internet:
            st.session_state.web_content = get_web_context(prompt, search_count, max_length)
            
            enhanced_prompt = (
                f"Question: {prompt}\n\n"
                f"Web Context (for reference):{st.session_state.web_content}\n\n"
                "Please provide a detailed answer using the above context when relevant."
            )
        else:
            enhanced_prompt = prompt

        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': enhanced_prompt}]
        )
        st.session_state.response = response['message']['content']
    except Exception as e:
        st.session_state.response = f"Error: {str(e)}"

def main():
    st.title("Ollama Search Assistant")
    
    # Settings sidebar
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
        use_internet = st.checkbox("Enable Internet Search", value=True)
        search_count = st.slider("Max Search Results", 1, 10, 3)
        max_length = st.slider("Max Content Length (chars)", 100, 10000, 2000)
        
        if st.button("Clear All"):
            st.session_state.web_content = ""
            st.session_state.response = ""
            st.session_state.processing = False
    
    # Main content
    prompt = st.text_input("Enter your question here:", key="prompt")
    
    col1, col2 = st.columns([1, 1])
    
    if st.button("Search", disabled=st.session_state.processing):
        if not prompt:
            st.warning("Please enter a question.")
        else:
            st.session_state.processing = True
            with st.spinner("Processing your request..."):
                generate_response(prompt, model, use_internet, search_count, max_length)
            st.session_state.processing = False
            st.rerun()
    
    with col1:
        st.subheader("Web Content")
        st.text_area("Web content", st.session_state.web_content, height=400, key="web_content_display", label_visibility="collapsed")
    
    with col2:
        st.subheader("Response")
        st.text_area("Response", st.session_state.response, height=400, key="response_display", label_visibility="collapsed")
    
    # Save button at the bottom
    if st.session_state.response:
        if st.download_button(
            label="Save Output",
            data=f"Question: {prompt}\n\nWeb Content:\n{st.session_state.web_content}\n\nAnswer:\n{st.session_state.response}",
            file_name="Ollama_Output.txt",
            mime="text/plain"
        ):
            st.toast("Output saved successfully!", icon="✅")

if __name__ == "__main__":
    main()
