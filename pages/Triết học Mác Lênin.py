import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve
from prompt import template

st.set_page_config(
    page_title="Triết học Mác Lênin Chatbot",
    page_icon="📘",
)

# Khởi tạo mô hình LLM
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Tiêu đề cho trang
st.title("Chatbot - MacLenin")

# Tạo ô nhập liệu cho câu hỏi
query = st.text_input("Nhập câu hỏi của bạn:")

# Khi người dùng nhập câu hỏi
if query:
    # Truy vấn và lấy kết quả từ RAG
    results = retrieve(query, 'maclenin.pdf')
    doc = "\n".join(results)  # Tạo văn bản từ kết quả truy vấn

    # Sử dụng model để trả lời câu hỏi
    response = chain.invoke({'doc': doc, 'query': query})

    # Hiển thị kết quả
    st.write("Câu trả lời từ chatbot:")
    st.write(response)
