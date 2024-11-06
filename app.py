import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
from transformers import T5Tokenizer, T5ForConditionalGeneration

def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def retrieve_passages(query, model, passage_embeddings, passages, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, passage_embeddings, top_k=top_k)
    retrieved_passages = [passages[hit['corpus_id']] for hit in hits[0]]
    return retrieved_passages

def generate_answer(query, passages, tokenizer, model):
    context = " ".join(passages)
    input_text = f"question: {query} context: {context}"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output_ids = model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return answer

# App title
st.title('Deep Learning Book Q&A')
st.write("Ask questions about the 'Deep Learning' book by Ian Goodfellow")

# File uploader
uploaded_file = st.file_uploader("Upload the 'Deep Learning' book PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from PDF..."):
        book_text = extract_text_from_pdf(uploaded_file)
        st.success("Text extracted successfully!")

    passages = book_text.split('\n\n')

    # Load Sentence-BERT model
    with st.spinner("Loading retrieval model..."):
        retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')
        passage_embeddings = retrieval_model.encode(passages, convert_to_tensor=True)
    
    # Load T5 model
    with st.spinner("Loading generation model..."):
        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        generation_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    
    query = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        with st.spinner("Retrieving passages..."):
            retrieved_passages = retrieve_passages(query, retrieval_model, passage_embeddings, passages)
        
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, retrieved_passages, tokenizer, generation_model)
        
        st.write(f"**Answer**: {answer}")
        st.write("**Retrieved Passages**:")
        for i, passage in enumerate(retrieved_passages):
            st.write(f"Passage {i+1}:\n{passage}\n")

# Run the app
if __name__ == '__main__':
    st.write("Upload the book and ask questions to get started!")
