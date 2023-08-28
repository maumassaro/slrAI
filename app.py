__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
import sqlite3

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks

# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_and_get_answer(vector_store, q, k=5):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    price = total_tokens / 1000 * 0.0004
    return total_tokens, price

if __name__ == "__main__":
    import os
    #from dotenv import load_dotenv, find_dotenv
    #load_dotenv(find_dotenv(), override=True)

    st.image('img.png')
    st.subheader('AI-Based Structured Literature Review Application')
    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type="password")
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key
        elif st.secrets["OPENAI_API_KEY"]:
            api_key = st.secrets["OPENAI_API_KEY"]
            os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
        
        upload_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512)
        k = st.number_input('K', min_value=1, max_value=20, value=5)
        add_data = st.button('Add Data')

        if upload_file and add_data:
            if api_key:
                with st.spinner('Reading, chunking and emedding file ...'):
                    bytes_data = upload_file.read()
                    file_name = os.path.join('./', upload_file.name)
                    with open(file_name, 'wb') as f:
                        f.write(bytes_data)

                    data = load_document(file_name)
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'Chunk ize: {chunk_size}, Chunks: {len(chunks)}')

                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    st.write(f'Embedding cost: {embedding_cost:.4f}')

                    vectore_store = create_embeddings(chunks)

                    st.session_state.vs = vectore_store
                    st.success('File uploaded, chunked and embedded successfully')
            else:
                st.warning('Please, add the api key and a file to proceed')

    q = st.text_input('Ask a question about the content of your file:')
    answer = ''
    if q and api_key:
        if ('vs' in st.session_state):
            vectore_store = st.session_state.vs
            answer = ask_and_get_answer(vectore_store, q, k)
            st.text_area('LLM Answer:', value=answer, height=400)
    else:
        Warning('Please, add the api key and a file to proceed')
    
    st.divider()
    if 'history' not in st.session_state:
        st.session_state.history = ''
    if answer:
        value = f'Q: {q} \nA: {answer}'
        st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
        h = st.session_state.history
        st.text_area(label=' Chat History', value = h, key = 'history', height=400)
    else:
        st.text('Chat history will be displayed here')
