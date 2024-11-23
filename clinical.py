import streamlit as st
import os
import numpy as np
import pandas as pd
import torch
import re
import cohere
from datasets import load_dataset
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Set the environment variables
os.environ['HF_HOME'] = '/path/to/your/hf/cache'  # Custom cache directory
os.environ['HUGGINGFACE_TOKEN'] = ''

# Log in to Hugging Face Hub
login(token=os.environ['HUGGINGFACE_TOKEN'])

API_KEY_COHERE = ''
co = cohere.Client(API_KEY_COHERE)

# Set Streamlit page configuration
st.set_page_config(
    page_title="Clinical Chatbot",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for dark theme
st.markdown("""
    <style>
        body {
            background-color: #000000;  /* Black background */
            color: #FFFFFF;  /* White text */
        }
        .chat-row {
            display: flex;
            margin: 10px 0;
        }
        .row-reverse {
            flex-direction: row-reverse;
        }
        .chat-bubble {
            padding: 10px;
            border-radius: 10px;
            max-width: 60%;
            word-wrap: break-word;
        }
        .ai-bubble {
            background-color: #FFFFFFFF;  /* Soft pink for AI messages */
            color: #000000FF;  /* White text */
        }
        .human-bubble {
            background-color: #FFFFFFFF;  /* Soft yellow for human messages */
            color: #000000;  /* Black text */
        }
        .stTextInput > div > input {
            background-color: #333333;  /* Dark grey input background */
            color: #FFFFFF;  /* White text */
            border: 1px solid #777777;  /* Light grey border */
            border-radius: 5px;  /* Rounded corners */
        }
        .stTextInput > div > input:focus {
            border-color: #F72843FF;  /* Soft pink focus border */
        }
        .stButton {
            background-color: #04FF00;  /* Bright green for button */
            color: white;
            border: none;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 5px;
            transition: background-color 0.3s;
        }
        .stButton:hover {
            background-color: #5E6400;  /* Darker green on hover */
        }
    </style>
""", unsafe_allow_html=True)

# Load the dataset
dataset = load_dataset("gamino/wiki_medical_terms")
docs_texts = dataset['train']['page_text']

# Function to split texts into chunks
def split_into_chunks(text_list, chunk_size=1):
    for i in range(0, len(text_list), chunk_size):
        yield ' '.join(text_list[i:i + chunk_size])

# Create chunks for Cohere
chunks = list(split_into_chunks(docs_texts, chunk_size=1))

# Load embeddings from CSV
embeddings_df = pd.read_csv('/content/drive/MyDrive/embeddings (1).csv')
embeddings = np.array(embeddings_df.values)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models and move to the appropriate device
@st.cache_resource
def load_models():
    pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
    pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum").to(device)
    llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    llama_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct").to(device)
    return pegasus_tokenizer, pegasus_model, llama_tokenizer, llama_model

if 'models' not in st.session_state:
    st.session_state.models = load_models()

pegasus_tokenizer, pegasus_model, llama_tokenizer, llama_model = st.session_state.models

# Function to chunk text for summarization
def chunk_text(text, max_length=512):
    sentences = text.split('. ')
    chunks = []
    current_chunk = []

    for sentence in sentences:
        current_chunk.append(sentence)
        if len('. '.join(current_chunk)) > max_length:
            chunks.append('. '.join(current_chunk[:-1]))
            current_chunk = [current_chunk[-1]]  # Start a new chunk with the last sentence

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append('. '.join(current_chunk))

    return chunks

# Function to summarize text using Pegasus
def summarize_text(text, max_length=200):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        inputs = pegasus_tokenizer(chunk, truncation=True, return_tensors="pt", max_length=1024).to(device)
        summary_ids = pegasus_model.generate(
            inputs['input_ids'],
            max_length=200,
            num_beams=5,
            length_penalty=2.0,
            early_stopping=True
        )
        summary = pegasus_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Combine summaries
    final_summary = ' '.join(summaries)

    # Optional: Trim the final summary to ensure it's under 200 words
    final_summary_words = final_summary.split()
    if len(final_summary_words) > 200:
        final_summary = ' '.join(final_summary_words[:200])  # Keep only the first 200 words

    return final_summary

# Function to generate answers using Llama
def generate_answer(query, context):
    combined_input = (
        f"Please read the following context and answer the question based on it:\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:"
    )
    
    inputs = llama_tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    outputs = llama_model.generate(
        inputs['input_ids'],
        max_length=400,
        num_beams=2,
        do_sample=True,
        top_k=100,
        temperature=0.7,
        early_stopping=True
    )
    
    generated_answer = llama_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_answer(generated_answer)

# Function to extract the answer from the AI's response
def extract_answer(ai_response):
    match = re.search(r'Answer:\s*(.*)', ai_response, re.DOTALL)  # Regex to find the answer
    return match.group(1).strip() if match else "I'm sorry, I couldn't find an answer."

# Function to perform similarity search using embeddings
def search_embeddings(query, number_of_results=3):
    # Placeholder for actual embedding logic
    query_embedding = np.random.rand(1, embeddings.shape[1])  
    distances = np.linalg.norm(embeddings - query_embedding, axis=1)
    closest_indices = distances.argsort()[:number_of_results]
    context = "\n".join(chunks[i] for i in closest_indices)
    summarized_context = summarize_text(context)
    final_answer = generate_answer(query, summarized_context)
    return final_answer

# Chatbot logic
if 'history' not in st.session_state:
    st.session_state.history = []

# Display chat history
for chat in st.session_state.history:
    div = f"""
    <div class="chat-row {'row-reverse' if chat['origin'] == 'human' else ''}">
        <div class="chat-bubble {'ai-bubble' if chat['origin'] == 'ai' else 'human-bubble'}">
            &#8203;{chat['message']}
        </div>
    </div>
    """
    st.markdown(div, unsafe_allow_html=True)

# Interface for user input
with st.form(key='chat_form'):
    st.markdown("**Chat**")
    user_input = st.text_input("Ask clinical questions:", key="human_prompt")

    # Submit button
    submit_button = st.form_submit_button("Submit")

if submit_button:
    if user_input:
        with st.spinner("Waiting for response..."):
            response_text = search_embeddings(user_input)
        st.session_state.history.append({'origin': 'human', 'message': user_input})
        st.session_state.history.append({'origin': 'ai', 'message': response_text})
