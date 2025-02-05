import streamlit as st
import pickle
import docx
import PyPDF2
import re

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('clf.pkl', 'rb'))  # Adjust the filename if needed
tfidf = pickle.load(open('tfidf.pkl', 'rb'))  # Adjust the filename if needed
le = pickle.load(open('encoder.pkl', 'rb'))  # Adjust the filename if needed


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


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


# Function to extract text from TXT
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


# Experience extraction function
def extract_experience(resume_text):
    experience_pattern = r'(\d+[\+\-]?\s?years?|\bover\s\d+\syears?)'
    match = re.search(experience_pattern, resume_text.lower())
    if match:
        experience_str = match.group(0)
        experience = int(re.sub(r'\D', '', experience_str))
        return experience
    return 0


# Evaluate resume function
def evaluate_resume(resume_text):
    score = 0

    # Step 1: Extract Experience
    experience = extract_experience(resume_text)
    score += experience * 5  # Experience contributes 5 points per year

    # Step 2: Check for Relevant Skills (Languages, Frameworks, Tools, etc.)
    skills_keywords = ['python', 'java', 'c++', 'react', 'node.js', 'angular', 'sql', 'git', 'docker', 'devops', 'html',
                       'css', 'typescript', 'aws', 'azure', 'kubernetes', 'google cloud', 'spring', 'flutter', 'ruby',
                       'swift', 'scala', 'vagrant', 'gitlab', 'jenkins', 'graphql']
    for skill in skills_keywords:
        if skill.lower() in resume_text.lower():
            score += 10  # Each matching skill contributes 10 points

    # Step 3: Check for Projects
    project_keywords = ['web development', 'mobile app', 'machine learning', 'cloud computing', 'devops', 'blockchain',
                        'full-stack', 'ai', 'data science', 'big data', 'deep learning', 'iot']
    if any(project in resume_text.lower() for project in project_keywords):
        score += 5  # Adding points for mentioning relevant projects

    # Step 4: Check for Certifications
    certifications_keywords = ['aws certified', 'azure certified', 'google cloud certified', 'certified kubernetes',
                               'certified scrum master', 'ccna', 'aws solutions architect', 'certified ethical hacker',
                               'oracle certified']
    if any(cert in resume_text.lower() for cert in certifications_keywords):
        score += 5  # Adding points for relevant certifications

    # Step 5: Check for relevant tech stack
    tech_stack_keywords = ['react', 'node.js', 'angular', 'spring', 'docker', 'kubernetes', 'aws', 'azure',
                           'google cloud', 'devops', 'graphql', 'machine learning', 'data science', 'sql', 'python',
                           'java']
    if any(tech_stack in resume_text.lower() for tech_stack in tech_stack_keywords):
        score += 5  # Adding points for tech stack

    # Final Decision
    if score >= 60:
        return "Applicable for Software Engineer role", score
    else:
        return "Not applicable for Software Engineer role", score


# Rank multiple resumes function
def rank_resumes(resumes):
    resume_scores = []
    for resume in resumes:
        result, score = evaluate_resume(resume)
        resume_scores.append((resume, result, score))

    ranked_resumes = sorted(resume_scores, key=lambda x: x[2], reverse=True)
    return ranked_resumes[:5]


# Streamlit page layout
def main():
    st.title('Resume Evaluator')

    # Upload button for single resume
    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        resume_text = handle_file_upload(uploaded_file)
        resume_text_clean = cleanResume(resume_text)

        result, score = evaluate_resume(resume_text_clean)
        st.write(f"**Result:** {result}")
        st.write(f"**Score:** {score}")

    # Multiple resume upload and ranking
    st.header("Upload Multiple Resumes to Rank Top 5")

    multi_files = st.file_uploader("Upload Resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if multi_files:
        resumes = []
        for uploaded_file in multi_files:
            resume_text = handle_file_upload(uploaded_file)
            resumes.append(cleanResume(resume_text))

        top_resumes = rank_resumes(resumes)
        st.write("Top 5 Resumes:")
        for i, (resume, result, score) in enumerate(top_resumes, 1):
            st.write(f"{i}. **Score:** {score}, **Result:** {result}")


# Ensure the script runs when executed
if __name__ == "__main__":
    main()

