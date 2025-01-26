from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

def query_gen(query, number):
    model = OllamaLLM(model='llama3')
    template = '''
Bạn là một công cụ mở rộng truy vấn được thiết kế để cải thiện độ chính xác của tìm kiếm bằng cách tạo ra {number} phiên bản mở rộng của truy vấn gốc từ người dùng.

**Quy định bắt buộc**
- Chỉ tạo đúng {number} phiên bản mở rộng.
- Mỗi phiên bản là một câu duy nhất, viết bằng tiếng Việt, ngắn gọn, súc tích.
- Không giải thích, diễn giải, hoặc thêm bất kỳ nội dung nào khác ngoài các câu truy vấn.
- Nếu không thể tạo đủ {number} câu truy vấn, hãy trả về thông báo lỗi: "Không thể tạo đủ số lượng truy vấn mở rộng."
- Không thêm bất kỳ tiêu đề, câu dẫn, hoặc giải thích nào như "Here are the expanded queries:".

Đây là truy vấn gốc: {query}

Ví dụ truy vấn gốc: Tư tưởng Hồ Chí Minh là gì
Ví dụ phản hồi (với số phiên bản = 3):
Tư tưởng Hồ Chí Minh là quan điểm lý luận về cuộc sống và xã hội.
Phái đoàn tư tưởng Hồ Chí Minh và ảnh hưởng của nó đến lịch sử Việt Nam.
Lý thuyết tư tưởng Hồ Chí Minh và vai trò trong xây dựng chủ nghĩa xã hội tại Việt Nam.

Hãy chính xác và cẩn thận. Không được lệch khỏi định dạng yêu cầu.
Chỉ trả lời duy nhất {number} dòng kết quả.

'''
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    result = chain.invoke({"number": number, "query": query})
    lines = result.strip().split("\n")
    results = lines[len(lines)-number :]
    print(lines)
    results.append(query)  # Append the original query to the result list
    print(results)
    return results  # Return the cleaned list

