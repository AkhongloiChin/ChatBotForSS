import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from prompt import template
from retriever import retrieve

# Khởi tạo mô hình LLM
model = OllamaLLM(model = 'llama3')
prompt = ChatPromptTemplate.from_template(template)
chain = prompt|model

# Tạo giao diện Streamlit
st.title("Chatbot Tư tưởng Hồ Chí Minh")

# Tạo ô nhập liệu cho câu hỏi
query = st.text_input("Nhập câu hỏi của bạn:")

# Khi người dùng nhập câu hỏi
if query:
    # Truy vấn và lấy kết quả từ RAG
    results = retrieve(query)
    doc = "\n".join(results)  # Tạo văn bản từ kết quả truy vấn

    # Sử dụng model để trả lời câu hỏi
    response = chain.invoke({'doc' : doc, 'query' : query})

    # Hiển thị kết quả
    st.write("Câu trả lời từ chatbot:")
    st.write(response)
