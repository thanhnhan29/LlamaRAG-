from groq import Groq
import os
import requests
from pprint import pprint
from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np
import re
from datetime import datetime, timedelta
def get_date_30_days_ago():
    today = datetime.now()
    date_30_days_ago = today - timedelta(days=30)
    return date_30_days_ago

time = get_date_30_days_ago()

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
model = BertModel.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

def embedding(text:str):
    inputs = tokenizer(text, return_tensors="pt")
    # Rút trích đặc trưng
    with torch.no_grad():
        outputs = model(**inputs)

    # Lấy embedding của [CLS] token (đặc trưng của toàn bộ câu)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.numpy()
# completion = client.chat.completions.create(
#     model="llama3-70b-8192",
#     messages=[
#         {
#             "role": "user",
#             "content": "Trường đại học khoa học tự nhiên có bao nhiêu khoa"
#         },
#     ],
#     temperature=1,
#     max_tokens=1024,
#     top_p=1,
#     stream=False,
#     stop=None,
# )
# print(completion.choices[0].message.content

def findInfomation(question = None):
    keyword = generate45info(question)
    print("keyword", keyword)
    keyword = "%20".join(keyword.split(" "))
    link = f"https://newsapi.org/v2/everything?q={keyword}&from={time.year}-{time.month}-{time.day}&sortBy=publishedAt&apiKey=yourkey"
    link = re.sub(r'\"+', '', link)
    
    respone = requests.get(link)
    articles = respone.json()["articles"]
    print(len(articles))
    
    articles = [item for item in articles if item["title"] != '[Removed]']
    print(len(articles))
    if len(articles) == 0:
        return "None"
    info = ["Title: " + str(item["title"]) + ". Description: " + str(item["description"]) + ". PublishedAt: " + str(item["publishedAt"]) + "\n" for item in articles]
    #print(info[:5])
    embeddings = np.vstack([embedding(text) for text in info])
    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]  # Số chiều của embeddings
    index = faiss.IndexFlatL2(dimension)  # Sử dụng L2 (Euclidean Distance) cho tìm kiếm
    index.add(embeddings)  # Thêm embeddings vào index
    query_embedding = embedding(question).astype('float32')

    # Tìm kiếm văn bản tương tự nhất
    D, I = index.search(query_embedding, 5)  # Tìm kiếm 2 văn bản gần nhất
    # print("Indices of nearest texts:", I)
    # print("Distances of nearest texts:", D)

    # In kết quả
    res = ""
    for idx in I[0]:
        res += info[idx] + "."
        print(info[idx])
    return res

apikey = "yourkey"


def generate45info(question):
    client = Groq(api_key=apikey)
    completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "user",
            "content": f"""
            Bạn là một module trong một hệ thống lớn, chỉ trả lời đáp án, không chú thích gì thêm\n \
            Hãy rút trích các keyword của câu sau: '{question}\n\
            Để tìm kiếm các thông tin liên quan cho câu hỏi?\n\
            Điều kiện bắt buộc của ouput: \n \
            - phải ngắn gọn súc tích phù hợp để sreach các trang báo, Ngôn ngữ sử dụng phải là tiếng anh \n\
            - chỉ được là một câu \n
            - không được có những chú thích thêm\n
            - chỉ có một dòng \n \
            - không được là câu hỏi\n \
            

            Định dạng câu trả lời: \n \
            - \n \

            Ví dụ mẫu: "Đội nào vô địch Euro 2024"\n
            Câu trả lời: "Tây ban nha" 
            """
        },
    ],
    temperature=0.1,
    max_tokens=32,
    top_p=1,
    stream=False,
    stop=None,)
    return completion.choices[0].message.content

def generate(question):
    info = findInfomation(question)
    client = Groq(api_key=apikey)
    completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "user",
            "content": f"""
            Bạn là người trợ lý hỗ trợ, \
            Hãy tham khảo những thông tin dưới đây không có thì bỏ qua: \n\
            Information:{info}\n
            ====================\n

            Để trả lời câu hỏi sau đây:\n \
            
            ====================\n \
            Question:{question}\n  \
            ====================\n \
            Định dạng câu trả lời: 
            - Câu trả lời phải cùng ngôn ngữ với câu hỏi bên trên,\
            - Không được sử dụng các cụm từ: "thông tin được cung cấp", "thông tin trên", ... Hoặc những cụm từ liên quan \n\
            - Câu trả lời phải ngắn gọn, súc tích.\n \

            Câu hỏi mẫu: "Đội nào vô địch bóng đá nam Euro 2021?"\n \
            câu trả lời: được tuyển ."\n
            """
        },
    ],
    temperature=0.1,
    max_tokens=512,
    top_p=1,
    stream=False,
    stop=None,)
    return completion.choices[0].message.content
#findInfomation("Who is the champion Euro")