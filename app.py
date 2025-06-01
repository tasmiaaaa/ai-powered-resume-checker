import os
import re
import docx
import pickle
import json
import PyPDF2
import streamlit as st
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.generativeai as genai

# streamlit run app.py

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('models/svc_model.pkl', 'rb'))
tfidf = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
le = pickle.load(open('models/label_encoder.pkl', 'rb'))

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Function to clean resume text
def clean_resume(txt):
    clean_text = re.sub('http\S+\s', ' ', txt)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    return clean_text

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def predict(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = clean_resume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    # Return the category name
    return predicted_category_name[0]


# Parse the structured response from Gemini
def parse_gemini_response(response_text: str) -> Dict[str, Any]:
    try:
        # Clean the response text by removing any markdown formatting or extra whitespace
        cleaned_text = response_text.strip()
        # Remove markdown code blocks if present
        if cleaned_text.startswith('```json'):
            cleaned_text = cleaned_text[7:]  # Remove ```json
        if cleaned_text.startswith('```'):
            cleaned_text = cleaned_text[3:]  # Remove ```
        if cleaned_text.endswith('```'):
            cleaned_text = cleaned_text[:-3]  # Remove closing ```

        cleaned_text = cleaned_text.strip()

        # Parse the JSON
        response_dict = json.loads(cleaned_text)

        # Validate and extract data with proper fallbacks
        return {
            'score': int(response_dict.get('score', 0)),
            'strengths': response_dict.get('strengths', ["Zero strengths"]),
            'weaknesses': response_dict.get('weaknesses', ["No weaknesses!"]),
            'missing_skills': response_dict.get('missing_skills', ["No missing skills!"]),
            'suggestions': response_dict.get('suggestions', ["No suggestions!"]),
            'formatting_feedback': response_dict.get('formatting_feedback', ["No formatting feedback"]),
            'summary': response_dict.get('summary', "No summary available")
        }

    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON from Gemini response: {str(e)}")
        st.error(f"Raw response: {response_text[:500]}...")  # Show first 500 chars for debugging

        # Return a default structure to prevent crashes
        return {
            'score': 0,
            'strengths': ["Unable to parse response"],
            'weaknesses': ["Response parsing failed"],
            'missing_skills': [],
            'suggestions': ["Please try again"],
            'formatting_feedback': [],
            'summary': "Analysis failed due to parsing error"
        }
    except Exception as e:
        st.error(f"Unexpected error parsing response: {str(e)}")
        return {
            'score': 0,
            'strengths': [],
            'weaknesses': [],
            'missing_skills': [],
            'suggestions': [],
            'formatting_feedback': [],
            'summary': "Unexpected error occurred"
        }

# Function to analyze resume with Gemini API
def analyze_resume_with_gemini(resume_text: str, job_description: str) -> Dict[str, Any]:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
            You are an expert HR professional and resume analyst. Analyze the following resume against the provided job description if job description is empty then analysis the resume only
            
            Job Description:
            {job_description}
            
            Resume:
            {resume_text}
            
            Please provide your analysis as a valid JSON object with the following structure. Return ONLY the JSON object, no additional text or markdown:

            {{
                "score": [number between 0-100],
                "strengths": [
                    "strength 1",
                    "strength 2",
                    "strength 3"
                ],
                "weaknesses": [
                    "weakness 1",
                    "weakness 2",
                    "weakness 3"
                ],
                "missing_skills": [
                    "missing skill 1",
                    "missing skill 2"
                ],
                "suggestions": [
                    "suggestion 1",
                    "suggestion 2",
                    "suggestion 3",
                    "suggestion 4",
                    "suggestion 5"
                ],
                "formatting_feedback": [
                    "formatting comment 1",
                    "formatting comment 2"
                ],
                "summary": "4-5 sentences summarizing the overall analysis"
            }}
            """
        response = model.generate_content(prompt)

        if not response or not response.text:
            st.error("Empty response from Gemini API")
            return {
                'score': 0,
                'strengths': [],
                'weaknesses': [],
                'missing_skills': [],
                'suggestions': [],
                'formatting_feedback': [],
                'summary': "No response received from API"
            }

        return parse_gemini_response(response.text)

    except Exception as e:
        st.error(f"Error analyzing resume with Gemini: {str(e)}")
        return {
            'score': 0,
            'strengths': [],
            'weaknesses': [],
            'missing_skills': [],
            'suggestions': [],
            'formatting_feedback': [],
            'summary': f"Analysis failed: {str(e)}"
        }


# Alternative fallback function if JSON parsing continues to fail
def parse_gemini_response_fallback(response_text: str) -> Dict[str, Any]:
    """
    Fallback parser for when Gemini doesn't return proper JSON
    """
    try:
        # Initialize default structure
        result = {
            'score': 50,  # Default middle score
            'strengths': [],
            'weaknesses': [],
            'missing_skills': [],
            'suggestions': [],
            'formatting_feedback': [],
            'summary': ""
        }

        lines = response_text.split('\n')
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for section headers
            if 'SCORE' in line.upper() or 'score' in line.lower():
                # Extract score using regex
                import re
                score_match = re.search(r'(\d+)', line)
                if score_match:
                    result['score'] = int(score_match.group(1))
            elif 'STRENGTH' in line.upper():
                current_section = 'strengths'
            elif 'WEAKNESS' in line.upper():
                current_section = 'weaknesses'
            elif 'MISSING' in line.upper():
                current_section = 'missing_skills'
            elif 'SUGGESTION' in line.upper():
                current_section = 'suggestions'
            elif 'FORMAT' in line.upper():
                current_section = 'formatting_feedback'
            elif 'SUMMARY' in line.upper():
                current_section = 'summary'
            elif line.startswith('-') or line.startswith('•'):
                # This is a list item
                if current_section and current_section != 'summary':
                    item = line.lstrip('-•').strip()
                    if item:
                        result[current_section].append(item)
            else:
                # Regular text, might be part of summary
                if current_section == 'summary' and line:
                    result['summary'] += line + " "

        # Clean up summary
        result['summary'] = result['summary'].strip()
        if not result['summary']:
            result['summary'] = "Resume analysis completed"

        return result

    except Exception as e:
        st.error(f"Fallback parsing also failed: {str(e)}")
        return {
            'score': 0,
            'strengths': ["Parsing failed"],
            'weaknesses': ["Could not analyze resume"],
            'missing_skills': ["Could not find any"],
            'suggestions': ["Please try submitting again"],
            'formatting_feedback': ["Formatting looks good!"],
            'summary': "Analysis could not be completed due to technical issues"
        }


# Function to display score with color coding
def display_score(score: int):
    if score >= 80:
        color = "#00FF00"  # Green
        status = "Excellent"
    elif score >= 60:
        color = "#FFA500"  # Orange
        status = "Good"
    elif score >= 40:
        color = "#FFFF00"  # Yellow
        status = "Average"
    else:
        color = "#FF0000"  # Red
        status = "Needs Improvement"

    st.subheader(f"Score: {score}/100")
    st.markdown(
        f"<span style='color:{color}; font-weight:bold'>{status}</span>",
        unsafe_allow_html=True
    )


# Web app layout
def main():
    st.set_page_config(page_title="AI Powered Resume Checker", page_icon="📄", layout="wide")
    st.title("AI Powered Resume Checker")
    st.subheader("Developed by [Tasmia Hussain](https://github.com/tasmiaaaa) and [Shila Rani Deb Mitu](https://github.com/Mitu-Dev)")
    st.markdown("Upload a resume and job description to get AI-powered analysis and improvement suggestions.")

    st.subheader("Upload Resume")
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    st.subheader("Job Description")
    job_description = st.text_area(
        "Paste the job description here:",
        height=200,
        placeholder="Paste the complete job description including requirements, responsibilities, and qualifications..."
    )

    if uploaded_file is not None and job_description.strip():
        try:
            # Extract text from the uploaded file
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Successfully extracted text from the uploaded resume.")

            # Display extracted text (optional)
            with st.expander("View Extracted Resume Text"):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            col1, col2, col3= st.columns([2, 2, 1])

            with col1:
                st.subheader("Resume Analysis")
                category = predict(resume_text)
                st.markdown(
                    f"**Predicted Category:** <span style='color:#39FF14; font-weight:bold'>{category}</span>",
                    unsafe_allow_html=True
                )

            with col2:
                placeholder = st.empty()
                with placeholder.container():
                    # Center native spinner in a smaller box
                    st.markdown(
                        """
                        <div style='display: flex; justify-content: center; align-items: center; height: 25px;'>
                        """,
                        unsafe_allow_html=True
                    )
                    with st.spinner("Analyzing resume with AI..."):
                        analysis = analyze_resume_with_gemini(resume_text, job_description)

            with col3:
                placeholder = st.empty()
                if analysis:
                    st.session_state['analysis'] = analysis
                    # Replace spinner with score
                    with placeholder.container():
                        display_score(analysis['score'])


            # Display AI analysis results
            if 'analysis' in st.session_state:

                # Create tabs for different sections
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Strengths", "Areas to Improve","Missing Skills", "Suggestions", "Formatting Feedback","Summary"])

                with tab1:
                    st.subheader("Resume Strengths")
                    for i, strength in enumerate(analysis['strengths'], 1):
                        st.write(f"{i}. {strength}")

                with tab2:
                    st.subheader("Weaknesses")
                    for i, weakness in enumerate(analysis['weaknesses'], 1):
                        st.write(f"{i}. {weakness}")

                with tab3:
                    st.subheader("Missing Skills")
                    if analysis['missing_skills']:
                        for i, skill in enumerate(analysis['missing_skills'], 1):
                            st.write(f"{i}. {skill}")
                    else:
                        st.write("No missing skills detected!.")

                with tab4:
                    st.subheader("Improvement Suggestions")
                    for i, suggestion in enumerate(analysis['suggestions'], 1):
                        st.write(f"{i}. {suggestion}")

                with tab5:
                    st.subheader("Formatting Feedback")
                    for i, feedback in enumerate(analysis['formatting_feedback'], 1):
                        st.write(f"{i}. {feedback}")

                with tab6:
                    st.subheader("Overall Summary")
                    st.write(analysis['summary'])

        except Exception as e:
            st.error("❌ An error occurred while processing the resume or analyzing it.")
            st.exception(e)  # Optional: shows full traceback for debugging

    elif uploaded_file is not None and not job_description.strip():
        st.warning("⚠️ Please provide a job description to get AI-powered analysis.")
    elif not uploaded_file and job_description.strip():
        st.warning("⚠️ Please upload a resume file to proceed.")

    # Footer
    st.write("")
    st.subheader("More Information")

    # Instructions
    with st.expander("How to use"):
        st.markdown("""
        1. **Upload Resume**: Upload your resume in PDF, DOCX, or TXT format.
        2. **Add Job Description**: Paste the complete job description you're applying for.
        3. **Get Analysis**: Click "Analyze with AI" to get detailed feedback.
        4. **Review Results**: Check your score, strengths, weaknesses, and improvement suggestions.
        """)

    # Privacy Policy
    with st.expander("Privacy Policy"):
        st.markdown("""
        ### Privacy & Data Usage Policy

        - **Resume Safety**: We do **not** store, save, or share any uploaded resumes. Your resume is processed securely and only used temporarily during the session for analysis purposes.

        - **AI Usage**: This application uses **Google's Gemini API** to analyze the resume against the provided job description. The data is sent securely to the API, and no personally identifiable information is stored.

        - **Your Control**: All uploaded files and data remain **local to your session** and are discarded once the session ends.

        Your privacy is important to us. This tool is built for educational and career-enhancement purposes only.
        """)

    # About Developers
    with st.expander("About the Developers"):
        st.markdown("""
        ### Tasmia Hussain  
        - Student at **Metropolitan University** 
        - [GitHub](https://github.com/tasmiaaaa)

        ### Shila Rani Deb Mitu  
        - Student at **Metropolitan University**  
        - [GitHub](https://github.com/Mitu-Dev)
        """)

if __name__ == "__main__":
    main()
