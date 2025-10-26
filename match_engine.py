import streamlit as st
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize SentenceTransformer once
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Streamlit page config
st.set_page_config(
    page_title="JobMatch AI",
    page_icon="💼",
    layout="wide"  # Changed to wide for full-screen layout
)

# Custom CSS styling
st.markdown("""
<style>
body {
    background-color: #f5f7fa;
    color: #333;
    font-family: 'Segoe UI', sans-serif;
}
.main {
    padding: 2rem 4rem;
    max-width: 95%;
    margin: auto;
}
.result-container {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    justify-content: center;
    width: 100%;
}
.result-box {
    background: linear-gradient(145deg, #e3f2fd, #bbdefb);
    padding: 25px;
    border-radius: 14px;
    margin-top: 20px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    width: 100%;
    line-height: 1.6;
}
.result-title {
    font-weight: 700;
    color: #0b3d91;
    font-size: 20px;
    margin-bottom: 10px;
}
.result-content {
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Compute semantic match score
def compute_match_score(resume_text, job_text):
    emb_resume = sbert_model.encode(resume_text, convert_to_tensor=True)
    emb_job = sbert_model.encode(job_text, convert_to_tensor=True)
    score = util.pytorch_cos_sim(emb_resume, emb_job).item()
    return round(score * 100, 2)

# Generate AI feedback using Gemini
def generate_ai_feedback(resume_text, job_text):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return "❌ Gemini API key not found. Please set GEMINI_API_KEY in your .env file."

    genai.configure(api_key=gemini_api_key)
    prompt = f"""
You are an expert career coach. Compare the given resume and job description and provide detailed insights.

Output format:
- JOB MATCH SCORE: ...
- MISSING SKILLS: ...
- RECOMMENDATIONS FOR IMPROVING OR CREATING NEW RESUME: ...
- SUMMARY: ...
 
Resume:
{resume_text}

Job Description:
{job_text}
"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No response generated."
    except Exception as e:
        import traceback
        return f"⚠️ Error while generating feedback: {e}\n{traceback.format_exc()}"

# Display results only if resume_text and job_text are defined
try:

    semantic_score = compute_match_score(resume_text, job_text)
    ai_feedback = generate_ai_feedback(resume_text, job_text)

    st.markdown("<div class='result-container'>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='result-box'><div class='result-title'>Semantic Match Score:</div>"
        f"<div class='result-content'>{semantic_score}%</div></div>", 
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='result-box'><div class='result-title'>AI Feedback:</div>"
        f"<div class='result-content'>{ai_feedback.replace(chr(10), '<br>')}</div></div>", 
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

except NameError:
    pass  # Silently do nothing if resume_text or job_text are not defined

