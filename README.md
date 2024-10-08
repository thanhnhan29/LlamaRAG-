### General:

- Utilizing the LLaMA 3 free API (available at ***[this link](https://console.groq.com/)***).
- To address the limitations of LLMs in general and LLaMA in particular, I have enhanced the information update process by incorporating RAG.
  
---

### Retrieval Information:

- Employing the News API to retrieve news related to the query (available at ***[this link](https://newsapi.org/)***).
- Using BERT to encode text into embeddings and then using Faiss to store and retrieve vectors quickly.

---

### Generation:

- Selecting the K-best articles to assist LLMs in generating accurate answers to questions.

---

### Results achieved:

- Llama 3 is now able to answer questions about recent issues, such as the results of the 2024 Olympics.
  #### Example
  ![Example](https://github.com/thanhnhan29/LlamaRAG-/blob/main/images/Example.png)
  #### Example 2
  ![Example](https://github.com/thanhnhan29/LlamaRAG-/blob/main/images/Example2.png)

---

### Demo:
- I use Streamlit framework to deloy this. You can find the link ***[here](https://llamaragapp-s6jcfgswm4fkzjiw5grfpp.streamlit.app/)***

### Next improvement focus:

- Developing a fully functional chat box that can remember chat history and respond to questions related to previous conversations
