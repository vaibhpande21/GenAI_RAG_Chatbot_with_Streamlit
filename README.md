# 📚 GenAI RAG Chatbot with Streamlit  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)  
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-brightgreen)](https://streamlit.io/)  
[![LangChain](https://img.shields.io/badge/Powered%20By-LangChain-orange)](https://www.langchain.com/)  
[![OpenAI](https://img.shields.io/badge/LLM-OpenAI%20GPT-red)](https://platform.openai.com/)  

---

## 🚀 Project Overview  
This project is a **Retrieval-Augmented Generation (RAG) Chatbot** built with **Streamlit**, **LangChain**, and **OpenAI GPT models**.  

It allows users to upload PDF documents, create vector embeddings, and then **ask natural language questions** to retrieve answers directly from the documents.  

✨ **Key Features**  
- 📂 Upload PDF files for context  
- 🧠 Vector embeddings powered by FAISS  
- 💬 Chat interface for interactive Q&A  
- ⚡ Retrieval-Augmented Generation for precise answers  
- 🎨 User-friendly Streamlit UI  

---

## 🛠️ Tech Stack  
- **Frontend/UI**: [Streamlit](https://streamlit.io/)  
- **LLM**: OpenAI GPT (via `langchain_openai`)  
- **Embeddings & Retrieval**: FAISS + LangChain  
- **Document Parsing**: PyPDF  
- **Environment Management**: Python `venv`  

---

## 📂 Project Structure  
```plaintext
genai-rag-chatbot/  
│── app.py              # Streamlit chatbot app  
│── requirements.txt    # Python dependencies  
│── .env.example        # Sample environment variables   
│── README.md           # Project documentation  
│── LICENSE             # Open-source license (MIT recommended)
```
---

## ⚙️ Setup & Installation


1️⃣ Clone the repository
```bash
git clone https://github.com/vaibhavpande21/genai-rag-chatbot.git
cd genai-rag-chatbot
```

2️⃣ Create a virtual environment
```bash
python -m venv .venv  
source .venv/bin/activate   # Mac/Linux  
.venv\Scripts\activate      # Windows
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Set up environment variables
```bash
Create a .env file in the project root:
OPENAI_API_KEY=openai_api_key
```

▶️ Running the App
Start the Streamlit app:
```bash
streamlit run app.py
```
The app will open in your browser at:
```bash
👉 http://localhost:8501
```

📖 How It Works:

1. Upload a PDF file.
2. The app extracts text, chunks it, and creates vector embeddings with FAISS.
3. When you enter a question, the system:
   a.) Retrieves the most relevant chunks.
   b.) Passes them as context to the OpenAI LLM.
   c.) Generates a precise, context-aware answer.
