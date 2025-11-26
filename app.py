from QUBE import generate_book, clean_output, save_as_word, docx_to_pdf, TranscriptExtracter,build_vectorstore,create_rag_chain
from QUERY import load_book_text,build_book_retriever,create_qa_chain,answer_question
from LIVESTREAM import process_audio_query
import os
import streamlit as st
import sounddevice as sd
import numpy as np

MASTER_PROMPT="""
You are a senior non-fiction book author, editor, and writing strategist with 20+ years of experience.

Your task is to take the user-provided YouTube transcript and transform it into a complete,
well-structured, professionally written non-fiction BOOK while keeping the content of the transcript.

Follow these instructions EXACTLY:

------------------
### 1. BOOK STRUCTURE
Create a full-length book with:

- A compelling **book title** written with a heading format: Title: [Title]
- A relevant **subtitle** written with a heading Subtitle
- A short but powerful **book description** with heading Book Description
- A **preface** section with heading Preface
- A detailed **table of contents** in form of bibliographic entries without writing Chapter instead just write the chapter title with proper numbering and sub-heading numbers inside an indendation of each chapter number and it should not contain the page number
- exclude any mention of "conclusion" or "resource" in the table of contents
- 8–12 **chapters**, depending on transcript depth
- Each chapter must include:
  - A chapter title in the form "Chapter X: [Title]"
  - An engaging introduction with heading "Introduction"
  - Multiple sub-sections with headings in format "Sub‑section Chapter X.Y: [Sub-section Title]"
  - Examples, stories, analogies, or metaphors (even if not in transcript)
  - Clear explanations & insights

- A **final conclusion** with heading Conclusion
- A **resources or further reading** section (optional) wiht heading Resources or Further Reading

------------------
### 2. WRITING STYLE GUIDELINES
- Write in **clear, engaging, beginner-friendly** language.
- Maintain the speaker’s original message but rewrite in book-quality prose.
- Remove filler words, repetition, and irrelevant content.
- Expand ideas logically — it should read like a book, not a transcript.
- Add explanation where needed so readers understand even without watching the video.
- Improve flow, transitions, examples, and storytelling.
- Make the tone **professional**, **insightful**, and **engaging**.

------------------
### 3. CONTENT TRANSFORMATION RULES
- Reorganize ideas from the transcript into a logical book structure.
- Combine scattered points into coherent chapters.
- Expand short ideas into complete explanations.
- Add new insights where necessary (but stay faithful to the topic).
- Remove timestamps, speaker tags, YouTube artifacts, or formatting.
- Fix grammar, clarity, and structure.
- The should be no space between chapter ending and next chapter heading.


Do NOT format paragraphs using bullet points, hyphens, or list markers. 
Write in normal paragraph style, not as a list. 
No sentence should begin with "-" or "*" or "•" unless explicitly asked.


------------------
### 4. QUALITY REQUIREMENTS
The final result must be:

- Fully polished and ready for publishing
- Human-like, rich, and detailed
- Long enough to feel like a proper book (8–25k words depending on model)
- Coherent, structured, and logically flowing
- Free from transcript-style repetitions

------------------
### 5. CONTEXT
Here is the transcript context. Use it deeply to construct the book:

{context}


Return ONLY the final book. Do not explain your process.

    """
# -----------------------------
# Sidebar
# -----------------------------
st.set_page_config(page_title="QUBE AI", layout="wide" )

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown(
    """
<style>

/* Full page dark red background */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"], 
.main, .appview-container, .block-container {
    background-color: #600000 !important;
    color: white !important;
}

/* Sidebar background (black) */
[data-testid="stSidebar"] {
    background-color: #000000 !important;
    color: white !important;
}

/* Buttons */
.stButton > button {
    background-color: #000000 !important;
    color: white !important;
    border-radius: 8px;
    padding: 0.5em 1em;
    font-size: 16px;
    font-weight: bold;
    border: 1px solid #333;
}

.stButton > button:hover {
    background-color: #121212 !important;
}

/* Inputs */
.stTextInput > div > div > input {
    background-color: #FFE5E5 !important;
    color: #000000 !important;
    border-radius: 5px;
}

::placeholder {
    color: #000000 !important;
    opacity: 1;
}

/* Download buttons */
.stDownloadButton > button {
    background-color: #FF0000 !important;
    color: white !important;
    border-radius: 6px;
    padding: 0.5em 1em;
    font-weight: bold;
}

/* Google Font */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* Sidebar font */
[data-testid="stSidebar"] * {
    font-family: 'Poppins', sans-serif !important;
    letter-spacing: 0.3px;
}

/* Sidebar Title */
.sidebar-title {
    font-size: 28px;
    font-weight: 700;
    color: #FF0000;
    text-shadow: 0 0 12px rgba(255,0,0,0.7);
    margin-bottom: -5px;
}

/* Sidebar Subtitle */
.sidebar-subtitle {
    font-size: 13px;
    color: #CCCCCC;
    margin-bottom: 20px;
}

/* Divider */
.sidebar-divider {
    height: 1px;
    background: linear-gradient(to right, #FF0000, #550000);
    margin: 15px 0;
    border-radius: 5px;
}

/* Navigation Label */
.nav-label {
    font-size: 14px;
    color: #AAAAAA;
    font-weight: 600;
    margin-bottom: 6px;
}

/* Footer */
.sidebar-footer {
    margin-top: 30px;
    font-size: 12px;
    color: #777777;
    text-align: center;
}

</style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------
# SIDEBAR CONTENT
# ------------------------------------
st.sidebar.markdown(
    """
    <div class="sidebar-title">QUBE AI</div>
    <div class="sidebar-subtitle">Query-Augmented YouTube Book Generating Engine</div>

    <div class="sidebar-divider"></div>

    <div class="nav-label">📌 Choose a Page</div>
    """,
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    ["📘 Book Generator", "💬 Textual Query Bot", "🎙️ Live Streaming Voice Bot"]
)

st.sidebar.markdown(
    """
    <div class="sidebar-divider"></div>
    <div class="sidebar-footer">
        Built by <b>KARAN MOHAN</b>
    </div>
    """,
    unsafe_allow_html=True
)

# Session state
# -----------------------------
if "docx_file" not in st.session_state:
    st.session_state.docx_file = None
if "pdf_file" not in st.session_state:
    st.session_state.pdf_file = None

# -----------------------------
# Page 1: Generate Book
# -----------------------------
if page == "📘 Book Generator":
    st.markdown("## 🎬 YouTube Transcript → AI Book Generator")
    st.markdown("Enter a YouTube URL to generate your professional book.")

    with st.container():
        youtube_url = st.text_input("YouTube URL", placeholder="Paste URL here...")
        generate_button = st.button("🚀 Generate Book")

    # Dynamic status box
    status_box = st.empty()

    if generate_button:
        if not youtube_url:
            st.error("Please enter a valid YouTube URL.")
        else:
            try:
                # Step 1: Extract transcript
                status_box.info("🔍 Extracting transcript...")
                transcript_text = TranscriptExtracter(youtube_url)
                status_box.success("✅ Transcript extracted!")

                # Step 2: Generate book
                status_box.info("✍️ Generating book (AI)...")
                book_text = generate_book(transcript_text)
                status_box.success("📖 Book generated!")

                # Step 3: Save DOCX
                status_box.info("💾 Saving Word document...")
                st.session_state.docx_file = save_as_word(book_text)
                status_box.success("✅ DOCX saved!")

                # Step 4: Convert to PDF
                status_box.info("📄 Converting to PDF...")
                st.session_state.pdf_file = docx_to_pdf(st.session_state.docx_file)
                status_box.success("✅ PDF created!")

            except Exception as e:
                status_box.error(f"❌ Error: {e}")

    # Download buttons
    if st.session_state.docx_file and st.session_state.pdf_file:
        st.markdown("### Download your generated book:")
        col1, col2 = st.columns(2)
        with col1:
            with open(st.session_state.docx_file, "rb") as f:
                st.download_button(
                    "📄 Download DOCX",
                    data=f,
                    file_name="generated_book.docx",
                    use_container_width=True
                )
        with col2:
            with open(st.session_state.pdf_file, "rb") as f:
                st.download_button(
                    "📕 Download PDF",
                    data=f,
                    file_name="generated_book.pdf",
                    use_container_width=True
                )
    st.session_state.docx_file="generated_book.docx"
# -----------------------------
# Page 2: Textual Query Bot (continued)
# -----------------------------
elif page == "💬 Textual Query Bot":
    st.markdown("## 🤖 Book Query Bot")
    st.markdown(
        "Ask any question about the book generated in Page 1, "
        "and the AI will answer based on its content."
    )

    # Ensure a book is loaded
    if st.session_state.docx_file is None:
        st.warning("Please generate a book first on the 'Book Generator' page.")
    else:
        # Question input
        question = st.text_input("Enter your question:", placeholder="Type your question here...")
        ask_button = st.button("Ask")
        # Dynamic response box
        response_box = st.empty()

        if ask_button:
            if not question.strip():
                st.error("Please enter a question.")
            else:
                try:
                    response_box.info("🧠 Fetching answer from the book...")
                    answer = answer_question(book_file=st.session_state.docx_file, question=question)
                    response_box.success("✅ Answer retrieved!")
                    st.markdown("**Answer:**")
                    st.write(answer)
                except Exception as e:
                    response_box.error(f"❌ Error: {e}")

# -----------------------------
# Page 3: Live Streaming Voice Query Bot
# -----------------------------
elif page == "🎙️ Live Streaming Voice Bot":
    # 🎙️ Voice Query Bot
    st.markdown("## 🎙️ Voice Query Bot")
    st.markdown(
        "Ask a question about the generated book **using your voice**. "
        "I'll transcribe it, answer from the book, and speak the answer back to you."
    )

    # Make sure a book exists
    book_path = st.session_state.get("docx_file", None)
    if not book_path or not os.path.exists(book_path):
        st.warning("Please generate a book first on Page 1 (Book Generator).")
    else:
        # Recording + model settings
        duration = st.slider(
            "Recording duration (seconds)",
            min_value=2,
            max_value=10,
            value=4,
        )

        model_name = st.selectbox(
            "Whisper model size (bigger = slower but more accurate)",
            options=["tiny", "base", "small"],
            index=1,  # default to 'base'
        )

        if st.button("🎧 Record question and get spoken answer"):
            try:
                with st.spinner("Recording from microphone, then thinking..."):
                    result = process_audio_query(
                        book_file=book_path,
                        duration=duration,
                        model_name=model_name,
                    )
            except Exception as e:
                st.error(f"❌ Error during voice query: {e}")
            else:
                st.success("✅ Done! Here's what I understood and answered:")

                with st.expander("📌 View Transcript & Answer"):
                 st.markdown("### 🎤 Recognized Question")
                 st.write(result["question_text"])

                 st.markdown("### 📘 Answer from the Book")
                 st.write(result["answer_text"])


                # Play spoken answer
                st.markdown("**Spoken answer (TTS):**")
                with open(result["answer_audio_path"], "rb") as f_a:
                    answer_audio_bytes = f_a.read()
                st.audio(answer_audio_bytes, format="audio/mp3")

                # Download button for the answer audio
                st.download_button(
                    "⬇️ Download answer audio",
                    data=answer_audio_bytes,
                    file_name="book_answer.mp3",
                    use_container_width=True,
                )
