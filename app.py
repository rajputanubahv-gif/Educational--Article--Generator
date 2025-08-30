import os
import json
import streamlit as st
import google.generativeai as genai
from langchain.llms.base import LLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph
from reportlab.lib.units import inch
from datetime import datetime
# Added imports for voice feature
from gtts import gTTS
import base64
import tempfile

# ========================
# Gemini API Setup (Secure Method)
# ========================
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
except (KeyError, Exception) as e:
    st.error("CRITICAL ERROR: Your Google API key is not configured correctly.")
    st.error(
        "Please add it to your secrets.toml file locally, or as a secret if deploying on Streamlit Community Cloud."
    )
    st.stop()

# ========================
# Gemini LLM Wrapper (Updated to Support Voice)
# ========================
class GeminiLLM(LLM):
    def __init__(self, model: str = "gemini-1.5-flash-latest"):
        super().__init__()
        self._client = genai.GenerativeModel(model)

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop=None) -> str:
        try:
            response = self._client.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Gemini API Error: {e}")
            return f"⚠️ Gemini API Error: {e}"

    # New method for generating audio from text
    def generate_audio(self, text: str, lang: str = "en") -> str:
        try:
            tts = gTTS(text=text, lang=lang, slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                with open(fp.name, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                os.unlink(fp.name)
            return audio_b64
        except Exception as e:
            st.error(f"Text-to-Speech Error: {e}")
            return None

# ========================
# RAG Pipeline & Embedding Model
# ========================
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_rag_chain(text: str):
    embeddings = get_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = [Document(page_content=text)]
    splitted_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(splitted_docs, embeddings)
    retriever = vector_store.as_retriever()
    llm = GeminiLLM()
    return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# ========================
# Draft Manager (Updated to Support Voice Downloads)
# ========================
class DraftManager:
    def __init__(self, path="drafts.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w") as f: json.dump([], f)

    def load(self):
        try:
            with open(self.path, "r") as f: return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError): return []

    def save(self, title, content):
        drafts = self.load()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        drafts.insert(0, {"title": title, "content": content, "created_at": now})
        with open(self.path, "w") as f: json.dump(drafts, f, indent=2)
        st.session_state.active_draft_title = title
        st.session_state.active_draft_content = content

    def delete(self, index):
        drafts = self.load()
        if 0 <= index < len(drafts):
            drafts.pop(index)
            with open(self.path, "w") as f: json.dump(drafts, f, indent=2)

    def _to_pdf(self, title, content):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        width, height = letter
        styles = getSampleStyleSheet()
        safe_content = content.replace('<', '&lt;').replace('>', '&gt;').replace(os.linesep, '<br/>')
        p = Paragraph(f"<b>{title}</b><br/><br/>{safe_content}", styles['Normal'])
        p.wrapOn(c, width - 2 * inch, height - 2 * inch)
        p.drawOn(c, inch, height - inch - p.height)
        c.save()
        buffer.seek(0)
        return buffer

    def sidebar_history(self):
        st.sidebar.markdown("## 📚 Draft History")
        drafts = self.load()
        if not drafts:
            st.sidebar.info("No drafts saved yet.")
        else:
            for i, d in enumerate(drafts):
                expander_title = f"📄 {d['title']} ({d.get('created_at', 'No Date')})"
                with st.sidebar.expander(expander_title):
                    st.button("🎯 Set as Active", key=f"select_{i}", on_click=self.set_active, args=(d,))
                    pdf_bytes = self._to_pdf(d["title"], d["content"])
                    st.download_button("⬇️ PDF", pdf_bytes, file_name=f"{d['title']}.pdf", key=f"pdf_{i}")
                    # New: Download audio for the draft
                    llm = GeminiLLM()
                    audio_b64 = llm.generate_audio(d["content"])
                    if audio_b64:
                        st.download_button(
                            "🔊 Audio", 
                            data=base64.b64decode(audio_b64), 
                            file_name=f"{d['title']}.mp3", 
                            key=f"audio_{i}"
                        )
                    if st.button("🗑️", key=f"del_{i}", help="Delete Draft"):
                        self.delete(i)
                        st.rerun()
    
    def set_active(self, draft):
        st.session_state.active_draft_title = draft['title']
        st.session_state.active_draft_content = draft['content']
        st.toast(f"Active draft set to: '{draft['title']}'")

# ========================
# Tool Executor (Updated for Enhanced Features)
# ========================
class ToolExecutor:
    def __init__(self, draft_manager: DraftManager):
        self.draft_manager = draft_manager
        self.llm = GeminiLLM()

    def execute(self, intent: str, params: dict, save_draft=True):
        content = params.get("article_content")
        title = params.get("article_title", "Untitled")
        
        if intent == "GENERATE_ARTICLE":
            user_prompt = params.get("prompt")
            if not user_prompt: return "Prompt is missing.", ""
            
            final_prompt = f"""
            {user_prompt}
            ---
            After writing the article, please add a "Further Resources" section at the very end. This section must include:
            1. A list of 3-5 relevant book recommendations (Title by Author).
            2. A list of 3-5 high-quality URL links to articles or reputable educational websites on the topic.
            3. A list of 3-5 relevant educational YouTube video links with a title and the full, valid URL (e.g., https://www.youtube.com/watch?v=XXXXX).
            Ensure all YouTube links are real and accessible by searching for actual videos.
            Format this section clearly with markdown headings like '### 📚 Books & Articles' and '### 🎥 Recommended Videos'.
            """
            new_content = self.llm._call(final_prompt)
            new_title = user_prompt[:70].strip().replace('\n', ' ') + "..."
            if save_draft: self.draft_manager.save(new_title, new_content)
            return new_content, new_title
        
        if not content:
            return "No article content provided. Please generate a new article or set an active one from the sidebar.", ""

        if intent == "QUESTION_ANSWERING":
            rag_chain = build_rag_chain(content)
            question = params.get("prompt", "")
            result = rag_chain.run(question)
            return result, ""

        if intent == "FURTHER_READING":
            prompt = f"""
            Based on the following text, suggest resources for further reading and viewing. The section must include:
            1. A list of 3-5 relevant book recommendations (Title by Author).
            2. A list of 3-5 high-quality URL links to articles or reputable educational websites.
            3. A list of 3-5 relevant educational YouTube video links with a title and the full, valid URL (e.g., https://www.youtube.com/watch?v=XXXXX).
            Ensure all YouTube links are real, accessible, and sourced from a reliable search.
            Format the output clearly with markdown headings like '### 📚 Books & Articles' and '### 🎥 Recommended Videos'.
            ---
            Text to analyze:
            {content}
            """
            result = self.llm._call(prompt)
            return result, ""

        if intent == "SUMMARIZE":
            rag_chain = build_rag_chain(content)
            result = rag_chain.run("Provide a concise summary of the key points in this document.")
            return result, ""

        elif intent == "KEYWORD_EXTRACTION":
            rag_chain = build_rag_chain(content)
            result = rag_chain.run("Extract the 5-7 most important keywords or entities. List them with a brief one-sentence definition for each, based only on the text.")
            return result, ""

        elif intent == "READABILITY_ANALYSIS":
            prompt = f"Analyze the following text for readability. Provide an estimated Flesch-Kincaid Grade Level and three specific suggestions to improve clarity for a general audience. Text: {content}"
            result = self.llm._call(prompt)
            return result, ""

        elif intent == "TRANSLATE":
            lang = params.get("target_language", "Spanish")
            # New: Language codes for gTTS audio generation
            lang_codes = {
                "Spanish": "es", "Portuguese": "pt", "French": "fr", "German": "de",
                "Japanese": "ja", "Mandarin Chinese": "zh-CN", "Hindi": "hi", 
                "Arabic": "ar", "Russian": "ru", "Polish": "pl", "Urdu": "ur",
                "Tamil": "ta", "Telugu": "te", "Malayalam": "ml", "Italian": "it",
                "Korean": "ko", "Bengali": "bn", "Dutch": "nl", "Swedish": "sv"
            }
            prompt = f"Translate the following text accurately into {lang}:\n\n{content}"
            new_content = self.llm._call(prompt)
            new_title = f"Translation ({lang}) of '{title}'"
            if save_draft: self.draft_manager.save(new_title, new_content)
            # New: Generate audio for translated content
            audio_b64 = self.llm.generate_audio(new_content, lang=lang_codes.get(lang, "en"))
            if audio_b64:
                st.download_button(
                    "🔊 Audio (Translated)", 
                    data=base64.b64decode(audio_b64), 
                    file_name=f"{new_title}.mp3",
                    key=f"translate_audio_{lang}"
                )
            return new_content, new_title

        elif intent == "REPHRASE":
            tone = params.get("new_tone", "more professional")
            rag_chain = build_rag_chain(content)
            new_content = rag_chain.run(f"Rephrase this document in a '{tone}' tone, ensuring all facts remain accurate.")
            new_title = f"Rephrased ({tone}) version of '{title}'"
            if save_draft: self.draft_manager.save(new_title, new_content)
            return new_content, new_title

        elif intent == "SOCIAL_MEDIA":
            platform = params.get("platform", "LinkedIn Post")
            prompt = f"Repurpose the following article into a compelling {platform}. For Twitter, create a thread of 3-4 tweets. For LinkedIn, write a professional post with a strong hook. Article:\n\n{content}"
            new_content = self.llm._call(prompt)
            new_title = f"{platform} for '{title}'"
            if save_draft: self.draft_manager.save(new_title, new_content)
            return new_content, new_title

        elif intent == "PRESENTATION_OUTLINE":
            prompt = f"Create a presentation outline based on the article. Structure it with a title slide, 3-5 main topic slides with 3-4 bullet points each, and a concluding summary slide. Article:\n\n{content}"
            new_content = self.llm._call(prompt)
            new_title = f"Outline for '{title}'"
            if save_draft: self.draft_manager.save(new_title, new_content)
            return new_content, new_title

        elif intent == "GENERAL_CHAT":
            response = self.llm._call(params.get("prompt", ""))
            # New: Generate audio for general chat response
            audio_b64 = self.llm.generate_audio(response)
            if audio_b64:
                st.download_button(
                    "🔊 Audio Response", 
                    data=base64.b64decode(audio_b64), 
                    file_name="chat_response.mp3",
                    key="general_chat_audio"
                )
            return response, ""

        return f"Unknown intent: {intent}", ""

# ========================
# Intent Router (For Chatbot)
# ========================
def get_intent(user_prompt: str) -> dict:
    llm = GeminiLLM()
    tools_list = "[GENERATE_ARTICLE, SUMMARIZE, QUESTION_ANSWERING, KEYWORD_EXTRACTION, READABILITY_ANALYSIS, TRANSLATE, REPHRASE, SOCIAL_MEDIA, PRESENTATION_OUTLINE, FURTHER_READING, GENERAL_CHAT]"
    router_prompt = f"""
    You are a smart routing agent. Based on the user's request, identify the correct tool from the list {tools_list} and its parameters. Respond in strict JSON format.

    - If the user asks to write, create, or generate something new, use `GENERATE_ARTICLE`.
    - If the user asks a specific question about the content of the *current document* (e.g., "what does it say about...", "who is mentioned..."), use `QUESTION_ANSWERING`.
    - If the user asks a general knowledge question (e.g., "what is the capital of France?"), use `GENERAL_CHAT`.
    - For all other tasks like summarizing, translating, etc., use the corresponding tool.

    Examples:
    - "write an article about the roman empire" -> {{"intent": "GENERATE_ARTICLE", "parameters": {{"prompt": "write an article about the roman empire"}}}}
    - "summarize the current article" -> {{"intent": "SUMMARIZE", "parameters": {{}}}}
    - "what does this article say about the main character?" -> {{"intent": "QUESTION_ANSWERING", "parameters": {{"prompt": "what does the article say about the main character?"}}}}
    - "suggest some books and videos on this topic" -> {{"intent": "FURTHER_READING", "parameters": {{}}}}
    - "what is the capital of France?" -> {{"intent": "GENERAL_CHAT", "parameters": {{"prompt": "what is the capital of France?"}}}}

    User Prompt: "{user_prompt}"
    """
    response_text = llm._call(router_prompt).strip()
    try:
        return json.loads(response_text.replace("```json", "").replace("```", "").strip())
    except json.JSONDecodeError:
        return {"intent": "GENERAL_CHAT", "parameters": {"prompt": user_prompt}}

# ========================
# Streamlit App
# ========================
st.set_page_config(page_title="AI Article Suite", layout="wide")
st.title("🎓 AI Article Suite")

draft_manager = DraftManager()
tool_executor = ToolExecutor(draft_manager)
draft_manager.sidebar_history()

def get_draft_selector(key_prefix: str):
    active_title = st.session_state.get("active_draft_title")
    drafts = draft_manager.load()
    if not drafts:
        st.warning("No drafts available. Please generate one first.")
        return None
    draft_titles = [d['title'] for d in drafts]
    try:
        current_index = draft_titles.index(active_title) if active_title in draft_titles else 0
    except ValueError:
        current_index = 0
    selected_title = st.selectbox(
        "Select a draft to work with (defaults to active draft):",
        draft_titles,
        index=current_index,
        key=f"selector_{key_prefix}"
    )
    return next((d for d in drafts if d['title'] == selected_title), None)

# --- Main Application Tabs ---
tab_titles = [
    "**💬 AI Chatbot**", "**✍️ Generate Article**", "**📊 Analyze & Summarize**", 
    "**🎨 Rephrase & Repurpose**", "**🖼️ Create Outline**", "**📚 Reading & Videos**", 
    "**🌐 Translate**", "**📂 Manage Drafts**"
]
tabs = st.tabs(tab_titles)

# --- Tab 1: AI Chatbot (Updated with Audio) ---
with tabs[0]:
    st.header("All-in-One AI Assistant")
    st.info("Ask me to generate a new article, summarize the active draft, or ask specific questions about its content!")

    active_title_display = st.session_state.get("active_draft_title")
    if active_title_display:
        st.success(f"**Active Draft:** '{active_title_display}'")
    else:
        st.warning("No active draft. Generate an article or select one from the sidebar to use content-specific features like 'summarize' or Q&A.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to do?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                action = get_intent(prompt)
                intent = action.get("intent")
                params = action.get("parameters", {"prompt": prompt})

                content_dependent_intents = [
                    "SUMMARIZE", "KEYWORD_EXTRACTION", "READABILITY_ANALYSIS", 
                    "TRANSLATE", "REPHRASE", "SOCIAL_MEDIA", 
                    "PRESENTATION_OUTLINE", "FURTHER_READING", "QUESTION_ANSWERING"
                ]

                active_content = st.session_state.get("active_draft_content")
                title_for_rerun = None

                if intent in content_dependent_intents and not active_content:
                    response = "⚠️ **Action requires an active article.**\nPlease generate a new article or select one from the sidebar history before using this feature."
                    st.warning(response)
                else:
                    if active_content:
                        params['article_title'] = st.session_state.get("active_draft_title")
                        params['article_content'] = active_content
                    
                    content, title_for_rerun = tool_executor.execute(intent, params)
                    
                    if intent == "GENERATE_ARTICLE" and title_for_rerun:
                        response = f"✅ **New article generated and saved as '{title_for_rerun}'!**\n\nIt is now the active draft. You can work with it using other tools or go to the **'📂 Manage Drafts'** tab to edit it.\n\n---\n\n{content}"
                    elif title_for_rerun:
                        response = f"✅ **New draft saved as '{title_for_rerun}'**\n\n---\n\n{content}"
                    else:
                        response = content
                    
                    st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        if title_for_rerun:
            st.rerun()

# --- Tab 2: Generate Article ---
with tabs[1]:
    st.header("Generate a New Article")
    st.info("Generates a complete article and automatically includes a 'Further Resources' section with books, articles, and videos.")
    full_prompt = st.text_area("Enter your prompt (topic, tone, style, etc.):", height=200, key="generate_prompt")
    if st.button("🚀 Generate Article & Resources", type="primary", key="generate_button"):
        if full_prompt.strip():
            with st.spinner("Generating..."):
                content, title = tool_executor.execute("GENERATE_ARTICLE", {"prompt": full_prompt})
                st.success(f"Successfully generated and saved '{title}'!")
                st.markdown("---")
                st.markdown(content)
                # New: Generate audio for the new article
                audio_b64 = tool_executor.llm.generate_audio(content)
                if audio_b64:
                    st.download_button(
                        "🔊 Audio", 
                        data=base64.b64decode(audio_b64), 
                        file_name=f"{title}.mp3",
                        key="generate_audio"
                    )
                st.rerun()
        else:
            st.error("Please enter a prompt.")

# --- Tab 3: Analyze & Summarize ---
with tabs[2]:
    st.header("Analyze an Existing Draft")
    selected_draft = get_draft_selector("analyze_tab")
    if selected_draft:
        sub_tabs = st.tabs(["**📊 Summarize**", "**🔑 Extract Keywords**", "**🔬 Readability**"])
        params = {"article_content": selected_draft['content'], "article_title": selected_draft['title']}
        with sub_tabs[0]:
            if st.button("Summarize Article", key="summarize_btn"):
                with st.spinner("Summarizing..."):
                    summary, _ = tool_executor.execute("SUMMARIZE", params, save_draft=False)
                    st.markdown(summary)
                    # New: Generate audio for the summary
                    audio_b64 = tool_executor.llm.generate_audio(summary)
                    if audio_b64:
                        st.download_button(
                            "🔊 Audio Summary", 
                            data=base64.b64decode(audio_b64), 
                            file_name=f"Summary_{selected_draft['title']}.mp3",
                            key="summary_audio"
                        )
        with sub_tabs[1]:
            if st.button("Extract Keywords", key="keywords_btn"):
                with st.spinner("Extracting..."):
                    keywords, _ = tool_executor.execute("KEYWORD_EXTRACTION", params, save_draft=False)
                    st.markdown(keywords)
                    # New: Generate audio for the keywords
                    audio_b64 = tool_executor.llm.generate_audio(keywords)
                    if audio_b64:
                        st.download_button(
                            "🔊 Audio Keywords", 
                            data=base64.b64decode(audio_b64), 
                            file_name=f"Keywords_{selected_draft['title']}.mp3",
                            key="keywords_audio"
                        )
        with sub_tabs[2]:
            if st.button("Analyze Readability", key="readability_btn"):
                with st.spinner("Analyzing..."):
                    analysis, _ = tool_executor.execute("READABILITY_ANALYSIS", params, save_draft=False)
                    st.markdown(analysis)
                    # New: Generate audio for the readability analysis
                    audio_b64 = tool_executor.llm.generate_audio(analysis)
                    if audio_b64:
                        st.download_button(
                            "🔊 Audio Readability", 
                            data=base64.b64decode(audio_b64), 
                            file_name=f"Readability_{selected_draft['title']}.mp3",
                            key="readability_audio"
                        )

# --- Tab 4: Rephrase & Repurpose ---
with tabs[3]:
    st.header("Rephrase or Repurpose a Draft")
    selected_draft = get_draft_selector("repurpose_tab")
    if selected_draft:
        params = {"article_content": selected_draft['content'], "article_title": selected_draft['title']}
        sub_tabs = st.tabs(["**🎨 Rephrase Tone**", "**📱 Create Social Post**"])
        with sub_tabs[0]:
            tone = st.selectbox("Select a new tone:", ["Formal", "Casual", "Professional", "Witty", "Simple"], key="rephrase_tone")
            if st.button("Rephrase in New Tone", key="rephrase_btn"):
                with st.spinner("Rephrasing..."):
                    params["new_tone"] = tone
                    content, title = tool_executor.execute("REPHRASE", params)
                    st.success(f"Rephrased and saved as '{title}'!")
                    st.text_area("Rephrased Content", content, height=250)
                    # New: Generate audio for the rephrased content
                    audio_b64 = tool_executor.llm.generate_audio(content)
                    if audio_b64:
                        st.download_button(
                            "🔊 Audio Rephrased", 
                            data=base64.b64decode(audio_b64), 
                            file_name=f"{title}.mp3",
                            key="rephrase_audio"
                        )
                    st.rerun()
        with sub_tabs[1]:
            platform = st.selectbox("Select social media platform:", ["LinkedIn Post", "Twitter (Tweet Thread)"], key="social_platform")
            if st.button("Generate Social Post", key="social_btn"):
                with st.spinner(f"Generating {platform}..."):
                    params["platform"] = platform
                    content, title = tool_executor.execute("SOCIAL_MEDIA", params)
                    st.success(f"Generated post and saved as '{title}'!")
                    st.text_area("Social Post Draft", content, height=250)
                    # New: Generate audio for the social media post
                    audio_b64 = tool_executor.llm.generate_audio(content)
                    if audio_b64:
                        st.download_button(
                            "🔊 Audio Social Post", 
                            data=base64.b64decode(audio_b64), 
                            file_name=f"{title}.mp3",
                            key="social_audio"
                        )
                    st.rerun()

# --- Tab 5: Create Outline ---
with tabs[4]:
    st.header("Create a Presentation Outline")
    selected_draft = get_draft_selector("outline_tab")
    if selected_draft:
        if st.button("🖼️ Generate Outline", key="outline_btn"):
            with st.spinner("Creating outline..."):
                params = {"article_content": selected_draft['content'], "article_title": selected_draft['title']}
                content, title = tool_executor.execute("PRESENTATION_OUTLINE", params)
                st.success(f"Outline created and saved as '{title}'!")
                st.text_area("Presentation Outline", content, height=400)
                # New: Generate audio for the outline
                audio_b64 = tool_executor.llm.generate_audio(content)
                if audio_b64:
                    st.download_button(
                        "🔊 Audio Outline", 
                        data=base64.b64decode(audio_b64), 
                        file_name=f"{title}.mp3",
                        key="outline_audio"
                    )
                st.rerun()

# --- Tab 6: Reading & Videos ---
with tabs[5]:
    st.header("Find Further Reading & Videos for a Draft")
    selected_draft = get_draft_selector("further_reading_tab")
    if selected_draft:
        if st.button("📚 Suggest Educational Resources", key="reading_btn"):
            with st.spinner("Searching for resources..."):
                params = {"article_content": selected_draft['content'], "article_title": selected_draft['title']}
                suggestions, _ = tool_executor.execute("FURTHER_READING", params, save_draft=False)
                st.markdown(suggestions)
                # New: Generate audio for the further reading suggestions
                audio_b64 = tool_executor.llm.generate_audio(suggestions)
                if audio_b64:
                    st.download_button(
                        "🔊 Audio Resources", 
                        data=base64.b64decode(audio_b64), 
                        file_name=f"Resources_{selected_draft['title']}.mp3",
                        key="resources_audio"
                    )

# --- Tab 7: Translate (Updated with Enhanced Languages and Audio) ---
with tabs[6]:
    st.header("Translate an Article")
    selected_draft = get_draft_selector("translate_tab")
    if selected_draft:
        # New: Expanded language options
        languages = [
            "Spanish", "Portuguese", "French", "German", "Japanese", "Mandarin Chinese",
            "Hindi", "Arabic", "Russian", "Polish", "Urdu", "Tamil", "Telugu", "Malayalam",
            "Italian", "Korean", "Bengali", "Dutch", "Swedish"
        ]
        target_language = st.selectbox("Translate to:", languages, key="translate_lang")
        if st.button("🌐 Translate", key="translate_btn"):
            with st.spinner(f"Translating to {target_language}..."):
                params = {"article_content": selected_draft['content'], "article_title": selected_draft['title'], "target_language": target_language}
                content, title = tool_executor.execute("TRANSLATE", params)
                st.success(f"Translated and saved as '{title}'!")
                st.text_area(f"Translated Text ({target_language})", content, height=300)
                st.rerun()

# --- Tab 8: Manage Drafts ---
with tabs[7]:
    st.subheader("Manage & Edit All Saved Drafts")
    drafts = draft_manager.load()
    if not drafts:
        st.info("No drafts to manage yet. Generate one from any other tab.")
    else:
        for idx, d in enumerate(drafts):
            with st.expander(f"📄 {d['title']} ({d.get('created_at', '')})"):
                new_content = st.text_area("Edit Draft", d["content"], key=f"edit_{idx}", height=250)
                if st.button("💾 Update Draft", key=f"upd_{idx}"):
                    drafts[idx]['content'] = new_content
                    with open(draft_manager.path, "w") as f: json.dump(drafts, f, indent=2)
                    st.success("Draft updated!")
                    st.rerun()