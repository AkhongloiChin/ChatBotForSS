from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
def query_gen(query, number):
    model = OllamaLLM(model='qwen2')
    template = '''
    Bạn là một công cụ mở rộng truy vấn được thiết kế để cải thiện độ chính xác của tìm kiếm bằng cách tạo ra {number} phiên bản mở rộng của truy vấn gốc từ người dùng.
    Đối với mỗi truy vấn mở rộng cần phải thỏa toàn bộ các yếu tố sau:
    -Từ đồng nghĩa hoặc các thuật ngữ tương tự nếu có (Hạn chế sử dụng những từ hoa mỹ , ưu tiên các từ mang tính chuyên môn mảng chính trị)
    -Các thuật ngữ hoặc cụm từ liên quan về ngữ cảnh.
    -Các cách diễn đạt khác nhưng vẫn giữ nguyên ý định ban đầu.

    Đây là truy vấn gốc: {query}
    Vui lòng tạo theo định dạng mỗi dòng một câu

    Các truy vấn mở rộng cần ngắn gọn, phù hợp và sử dụng cùng ngôn ngữ với truy vấn đầu vào. 
    Vì các câu truy vấn tạo ra không đúng nghĩa với câu truy vấn ban đầu sẽ gây ra tổn hại đến chất lượng hệ thống truy vấn, mong bạn hãy chính xác và cẩn thận
    '''
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    result = chain.invoke({"number": number, "query": query})
    lines = result.strip().split("\n")
    lines.append(query)  # Append the original query to the result list
    return lines  # Return the cleaned list
