from groq import Groq
import os
import requests
from pprint import pprint
from transformers import BertTokenizer, BertModel
import torch
import faiss
import numpy as np
import re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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
def embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # Sử dụng vector [CLS]
    return embeddings.numpy()


def findInfomation(question = None):

    print("\033c", end="")
    keyword = "%20".join(generate45info(question).split(" "))
    print("keyword", keyword)
    link = f"https://newsapi.org/v2/everything?q={keyword}&from=2024-07-10&sortBy=publishedAt&apiKey=<yourapikey>"
    link = re.sub(r'\"+', '', link)
    print(link)
    
    respone = requests.get(link)
    articles = respone.json()["articles"]
    print(len(articles))
    
    articles = [item for item in articles if item["title"] != '[Removed]']
    print(len(articles))
    if len(articles) == 0:
        return "None"
    info = [item["title"] + " " + item["description"] + " " + item["content"] for item in articles[:50]]
    #print(info[:5])
    embeddings = np.vstack([embedding(text, model, tokenizer) for text in info])
    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]  # Số chiều của embeddings
    index = faiss.IndexFlatL2(dimension)  # Sử dụng L2 (Euclidean Distance) cho tìm kiếm
    index.add(embeddings)  # Thêm embeddings vào index
    query_embedding = embedding(question, model, tokenizer).astype('float32')

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

apikey = "<yourapikey>"

def generate45info(question):
    client = Groq(api_key=apikey)
    completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "user",
            "content": f"""
            What's the keyword of '{question}' \
            to search for information to answer that question? \
            Answer briefly with only one phrase to indicate the answer, answer by English.
            """
        },
    ],
    temperature=0.1,
    max_tokens=64,
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
            "content": f"""You are an assistant.\
            Based on the information below and what you know, please answer the following question. \
            Respond in the language of the question and keep your answer brief and concise\n
            Information:{info}\n
            Question:{question}\n
            Do not use phrases like 'the provided information,' Do not use phrases such as 'the provided text,' \
            'the information provided.'or similar phrases.
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