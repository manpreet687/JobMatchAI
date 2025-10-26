import streamlit as st
import spacy
from nltk.corpus import stopwords
from PyPDF2 import PdfReader
from match_engine import compute_match_score, generate_ai_feedback

# Load NLP model and stopwords
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))


def extract_resume_data(text):
    """Extract basic info like name and skills from resume text."""
    doc = nlp(text)
    name = None
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    # Extract keywords as skills
    words = [token.text for token in doc if token.is_alpha and token.text.lower() not in stop_words]
    skills = list(set(words))[:30]

    return {"name": name, "skills": skills}


# -------------------- Streamlit UI --------------------
st.set_page_config(
    page_title="JobMatch AI – Smart Resume Matcher",
    page_icon="🧑‍💻",  # 👈 Tech Developer Icon
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
.main {
    padding: 2rem 4rem;
}
h1 {
    text-align: center;
    color: #0b3d91;
    font-size: 2.6rem !important;
    margin-bottom: 0.3rem;
}
.subtitle {
    text-align: center;
    font-size: 1.15rem;
    color: #555;
    margin-bottom: 2.5rem;
}
.upload-section {
    background-color: #eef5ff;
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.stButton>button {
    background-color: #0b3d91 !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: bold !important;
}
.stButton>button:hover {
    background-color: #0940b2 !important;
    }
   
</style>
""", unsafe_allow_html=True)

# Title Section
st.markdown("<h1>🧑‍💻 JobMatch AI-Smart Resume Matcher</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Get instant insights, match scores, and resume improvement tips</div>", unsafe_allow_html=True)

# Upload Resume Section
uploaded_resume = st.file_uploader("📄 Upload Resume (PDF or TXT)", type=["pdf", "txt"])
st.markdown("</div>", unsafe_allow_html=True)

# Job Description
job_description = st.text_area("💼 Paste Job Description Here", height=200)



# Analyze Button
if st.button("🚀 Analyze Resume"):
    if uploaded_resume and job_description:
        with st.spinner("Analyzing your resume... Please wait ⏳"):

            # Extract resume text
            if uploaded_resume.type == "text/plain":
                resume_text = uploaded_resume.getvalue().decode("utf-8")
            else:  # PDF extraction
                pdf = PdfReader(uploaded_resume)
                resume_text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])

            # Process and analyze
            resume_data = extract_resume_data(resume_text)
            resume_text_for_ai = str(resume_data)

            match_score = compute_match_score(resume_text_for_ai, job_description)
            feedback = generate_ai_feedback(resume_text_for_ai, job_description)

        # Output Section
        st.success("✅ Analysis Complete!")
        st.subheader(f"🎯 Match Score: {match_score}%")
        st.write("### 💡 AI Feedback and Suggestions")
        st.write(feedback)
        
    else:
        st.warning("⚠️ Please upload your resume and enter a job description.")
