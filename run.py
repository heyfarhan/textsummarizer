import streamlit as st 
from transformers import BartForConditionalGeneration, BartTokenizer
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from transformers import pipeline
import base64
import nltk
nltk.download('punkt')

from PyPDF2 import PdfReader



model_name = "facebook/bart-large-cnn"
base_model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

def file_preprocessing(file):
    reader = PdfReader(file)
    text=""
    for page in reader.pages :
        text=text + " " + page.extract_text()
    return text
    
def chunking(text):
    sentences = nltk.tokenize.sent_tokenize(text)
    length = 0
    chunk = ""
    chunks = []
    count = -1
    for sentence in sentences:
        count += 1
        combined_length = len(tokenizer.tokenize(sentence)) + length # add the no. of sentence tokens to the length counter

        if combined_length  <= tokenizer.max_len_single_sentence: # if it doesn't exceed
            chunk += sentence + " " # add the sentence to the chunk
            length = combined_length # update the length counter

            # if it is the last sentence
            if count == len(sentences) - 1:
                chunks.append(chunk.strip()) # save the chunk
            
        else: 
            chunks.append(chunk.strip()) # save the chunk
            length = 0 
            chunk = ""
            # take care of the overflow sentence
            chunk += sentence + " "
            length = len(tokenizer.tokenize(sentence))
        len(chunks)
    return chunks

def llm_pipeline(input_text,max_length):
    chunks = chunking(input_text)
    pipe_sum = pipeline(
        'summarization',
        model = base_model,
        tokenizer = tokenizer,
        max_length = int(max_length), 
        min_length = int(max_length-(int(max_length*0.3))))
    result=""
    for input in chunks:
        # output = base_model.generate(**input)
        output = pipe_sum(input)
        output = output[0]['summary_text']

        # print(output)
        result = result + " " + output
    return result




@st.cache_data
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

st.set_page_config(layout="wide")

def main():
    st.title("AI Summarization App")

    choice = st.sidebar.selectbox("Select your choice", ["Summarize Text", "Summarize Document"])
    max_length = st.sidebar.slider("Maximum Summary Length", min_value=0, max_value=500, value=150)


    if choice == "Summarize Text":
        input_text = st.text_area("Enter your text here")
        if input_text is not None:
            if st.button("Summarize Text"):
                col1, col2 = st.columns([1,1])
                with col1:
                    st.markdown("**Your Input Text**")
                    st.info(input_text)
                with col2:
                    st.markdown("**Summary Result**")
                    summary = llm_pipeline(input_text,max_length)
                    # summary = llm_pipeline(input_text)
                    st.info("Summarization Complete")
                    st.success(summary)
    elif choice == "Summarize Document":
        uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

        if uploaded_file is not None:
            if st.button("Summarize Text"):
                col1, col2 = st.columns(2)
                filepath = "data/"+uploaded_file.name
                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())
                with col1:
                    st.info("Uploaded File")
                    pdf_view = displayPDF(filepath)

                with col2:
                    input_text = file_preprocessing(filepath)
                    summary = llm_pipeline(input_text,max_length)
                    # summary = llm_pipeline(input_text)
                    st.info("Summarization Complete")
                    st.success(summary)



if __name__ == "__main__":
    main()