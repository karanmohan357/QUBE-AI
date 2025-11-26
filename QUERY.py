import langchain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_groq import ChatGroq
import os

def load_book_text(filename):
    from docx import Document
    doc = Document(filename)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip() != ""])
    return text

def build_book_retriever(text):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    chunks = [text[i:i + 1200] for i in range(0, len(text), 1200)]

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    return vectorstore.as_retriever()

def create_qa_chain(retriever):
    QA_PROMPT = """
    You are a helpful assistant. Use the provided book context to answer the question.

    Context:
    {context}

    Question:
    {question}

    Answer the question accurately, in complete sentences. If the answer is not in the context, say "The book does not provide an answer."
    """
    prompt = ChatPromptTemplate.from_template(QA_PROMPT)

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
        max_retries=3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return qa_chain

def create_qa_chain_voice_bot(retriever):
    QA_PROMPT = """
You are QUBE AI — a friendly, helpful AI assistant created by KARAN MOHAN.
You answer questions strictly using the book content provided. 
Do not use any outside knowledge.

Below is the book content you must use:
{context}

The user asked:
{question}

Your task:
- Answer ONLY using the information inside the book content.
- Never guess or fill missing details.
- Speak in a warm, natural, helpful AI assistant tone.
- Be conversational and easy to understand.
- Keep explanations clear and simple.
- No bullet points unless necessary.
- No robotic or repetitive phrases.

Assistant tone guidelines:
- Sound like a modern AI assistant (ChatGPT/Siri/Alexa style).
- Use friendly, human-like phrasing.
- Keep the response natural and supportive.
- Do not break character.

If the book does not contain the answer:
Say: “It looks like the book doesn’t provide information about this topic.”

Final Output Format:
Assistant:
<your friendly answer here>

    """

    prompt = ChatPromptTemplate.from_template(QA_PROMPT)

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.5,
        max_retries=3,
        api_key=os.getenv("GROQ_API_KEY")
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )

    return qa_chain


# def answer_question(book_file="generated_book.docx", question="Give a summary of the book."):
#     # Load book and build retriever
#     text = load_book_text(book_file)
#     retriever = build_book_retriever(text)
    
#     # Create QA chain
#     qa_chain = create_qa_chain(retriever)
    
#     # Correct: pass dict directly
#     input_vars = {
#         "context": text,
#         "question": question
#     }
    
#     # Invoke chain with dict
#     answer = qa_chain.invoke(input_vars)
#     return answer

def answer_question(book_file="generated_book.docx", question="Give a summary of the book."):
    text = load_book_text(book_file)
    retriever = build_book_retriever(text)
    qa_chain = create_qa_chain(retriever)

    # Pass only 'query' key or just the question string
    answer = qa_chain.run(question)  # simplest way
    return answer

def answer_question_voice_bot(book_file="generated_book.docx", question="Give a summary of the book."):
    text = load_book_text(book_file)
    retriever = build_book_retriever(text)
    qa_chain = create_qa_chain_voice_bot(retriever)

    # Pass only 'query' key or just the question string
    answer = qa_chain.run(question)  # simplest way
    return answer
