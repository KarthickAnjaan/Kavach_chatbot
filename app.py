import streamlit as st
from PyPDF2 import PdfReader
from PIL import Image
from io import BytesIO
import fitz  # PyMuPDF for extracting images from PDFs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai

# Streamlit app title
st.title("Kavach Guidelines Processor")

# Specify the local PDF file path (hardcoded for backend access only)
pdf_path = "Annexure-B.pdf"  # Ensure this file is in the same directory as this script

# Initialize session state to maintain chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Load and process PDF once at startup
@st.cache_resource
def load_pdf():
    pdf_reader = PdfReader(pdf_path)
    kavach_text = ''
    for page in pdf_reader.pages:
        kavach_text += page.extract_text() or ""  # Ensure handling of None
    return kavach_text

kavach_text = load_pdf()

# Step 4: Split the Kavach text into manageable chunks
chunk_size = 1000  # Number of characters in each chunk
chunk_overlap = 100  # Overlap between chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Step 5: Create text chunks and cache them
@st.cache_resource
def create_chunks(kavach_text):
    return text_splitter.split_text(kavach_text)

kavach_chunks = create_chunks(kavach_text)
kavach_chunks = [{'text': chunk, 'source': 'kavach_source'} for chunk in kavach_chunks]

# Step 6: Store the embeddings in a FAISS vector store and cache it
@st.cache_resource
def create_vector_store(kavach_chunks):
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_texts(
        texts=[chunk['text'] for chunk in kavach_chunks],
        embedding=embeddings,
        metadatas=[{'source': chunk['source']} for chunk in kavach_chunks]
    )

vectorstore = create_vector_store(kavach_chunks)

# Step 7: Configure Google Gemini Pro API (replace with your actual API key)
apiKey = "AIzaSyB44ykHvogjqMOlkF1Fi3pa4RfuNp3s9GA"  # Place your API key here directly for testing
genai.configure(api_key=apiKey)

def generate_kavach_response(prompt):
    model = genai.GenerativeModel('gemini-1.5-pro-002')
    response = model.generate_content(prompt)
    return response.text

# Step 8: Retrieve relevant documents and generate a response
def get_kavach_decision(query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    relevant_docs = retriever.get_relevant_documents(query)
    combined_text = "\n".join([doc.page_content for doc in relevant_docs])
    
    if not combined_text.strip():  # If no relevant documents found
        return "No relevant information found based on your query."
    
    prompt = f"Based on the following Kavach guidelines:\n\n{combined_text}\n\nAnswer the query: {query}"
    decision = generate_kavach_response(prompt)
    return decision

# User input for a text query
user_query = st.text_input("Enter your query about Kavach guidelines")

if st.button("Submit Query"):
    if user_query:
        # Generate the response based on the user's query
        decision = get_kavach_decision(user_query)

        # Store the query and response in session state
        st.session_state.chat_history.append({"user": user_query, "bot": decision})

        # Display the chat history
        st.header("Chat History")
        for chat in st.session_state.chat_history:
            st.write(f"**You**: {chat['user']}")
            st.write(f"**Bot**: {chat['bot']}")

        # Check if the query suggests image retrieval
        if "image" in user_query.lower() or "illustration" in user_query.lower():
            # Extract images from PDF if requested
            pdf_document = fitz.open(pdf_path)
            images = []

            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    image = Image.open(BytesIO(image_bytes))
                    images.append(image)

            # Display images if they are relevant
            if images:
                st.header("Extracted Images")

                # Create columns for displaying images side by side
                num_columns = 3  # Adjust the number of columns as needed
                cols = st.columns(num_columns)

                for i, image in enumerate(images):
                    # Resize the image to a consistent size (for example, 300x300)
                    resized_image = image.resize((300, 300), Image.LANCZOS)

                    # Use the column index to display the images
                    cols[i % num_columns].image(resized_image, caption=f"Image {i + 1}", use_column_width='auto')

                # Optional: If there are remaining images after filling columns, show them below
                if len(images) % num_columns != 0:
                    st.write("")  # Add a line break if needed
            else:
                st.write("No images found in the document.")
    else:
        st.warning("Please enter a query.")
