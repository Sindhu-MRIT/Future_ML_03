import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import nltk
import warnings
warnings.filterwarnings('ignore')

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="Resume Screener", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#f5f3ef}
div[data-testid="stSidebar"]{background:#1c1917}
div[data-testid="stSidebar"] *{color:#a8a29e !important}
div[data-testid="stSidebar"] hr{border-color:#292524;margin:14px 0}
div[data-testid="stSidebar"] .stSelectbox div{
    background:#292524 !important;border:1px solid #3c3836 !important;
    color:#a8a29e !important;border-radius:6px !important}
div[data-testid="stSidebar"] .stButton button{
    background:#d97706 !important;color:#fff !important;border:none !important;
    border-radius:6px !important;font-weight:600 !important;width:100% !important}
.headline{font-family:'Playfair Display',serif;font-size:48px;font-weight:700;
    color:#1c1917;line-height:1;margin-bottom:6px}
.subline{font-size:14px;color:#78716c;font-weight:300;margin-bottom:20px}
.hint-box{background:#fef3c7;border-radius:6px;padding:14px 18px;font-size:13px;
    color:#92400e;line-height:1.7;margin-bottom:24px}
.stat-card{background:#fff;border-radius:8px;padding:20px 18px;height:100%}
.stat-top{height:3px;border-radius:3px;margin-bottom:14px}
.stat-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#a8a29e;margin-bottom:6px}
.stat-num{font-size:28px;font-weight:600;color:#1c1917;line-height:1}
.stat-desc{font-size:12px;color:#a8a29e;margin-top:8px;line-height:1.6}
.section-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#d97706;margin-bottom:6px}
.section-title{font-family:'Playfair Display',serif;font-size:22px;color:#1c1917;margin-bottom:14px}
.candidate-card{background:#fff;border-radius:8px;padding:18px;margin-bottom:8px;
    border-left:4px solid #e7e5e4}
.candidate-card.excellent{border-left-color:#16a34a}
.candidate-card.good{border-left-color:#d97706}
.candidate-card.average{border-left-color:#dc2626}
.rank-num{font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#a8a29e;margin-bottom:4px}
.cand-score{font-size:24px;font-weight:600;color:#1c1917;line-height:1}
.cand-grade{display:inline-block;padding:3px 12px;border-radius:99px;font-size:11px;
    font-weight:600;margin-left:8px;vertical-align:middle}
.grade-excellent{background:#dcfce7;color:#166534}
.grade-good{background:#fef3c7;color:#92400e}
.grade-average{background:#fee2e2;color:#991b1b}
.skill-present{display:inline-block;background:#dcfce7;color:#166534;font-size:11px;
    padding:3px 10px;border-radius:99px;margin:2px;font-weight:500}
.skill-missing{display:inline-block;background:#fee2e2;color:#991b1b;font-size:11px;
    padding:3px 10px;border-radius:99px;margin:2px;font-weight:500}
.score-bar-wrap{height:6px;background:#e7e5e4;border-radius:3px;margin:8px 0 4px;flex:1}
.score-bar{height:100%;border-radius:3px}
.action-card{background:#1c1917;border-radius:8px;padding:20px;height:100%}
.ac-num{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#44403c;margin-bottom:8px}
.ac-title{font-size:14px;font-weight:600;color:#fafaf9;margin-bottom:8px}
.ac-body{font-size:13px;color:#57534e;line-height:1.8}
.ac-hi{color:#d97706;font-weight:500}
</style>
""", unsafe_allow_html=True)

STOP_WORDS = set(stopwords.words('english'))

JOB_DESCRIPTIONS = {
    'Information Technology': {
        'desc': 'python java sql software development programming algorithms data structures machine learning cloud aws docker linux agile scrum api rest database networking cybersecurity debugging testing git version control',
        'skills': ['python', 'java', 'sql', 'linux', 'aws', 'docker', 'git', 'api', 'database', 'networking', 'agile', 'cloud']
    },
    'Data Science': {
        'desc': 'python machine learning deep learning statistics data analysis pandas numpy scikit tensorflow pytorch sql tableau visualization feature engineering nlp regression classification clustering neural networks',
        'skills': ['python', 'machine learning', 'statistics', 'pandas', 'numpy', 'tensorflow', 'sql', 'visualization', 'nlp']
    },
    'Finance': {
        'desc': 'financial analysis accounting budgeting forecasting excel financial modeling investment portfolio risk management balance sheet income statement cash flow audit tax compliance regulations banking',
        'skills': ['excel', 'accounting', 'budgeting', 'forecasting', 'financial modeling', 'risk', 'audit', 'tax', 'compliance']
    },
    'Human Resources': {
        'desc': 'recruitment hiring onboarding employee relations payroll performance management training development hr policies talent acquisition workforce planning benefits compensation labor law',
        'skills': ['recruitment', 'onboarding', 'payroll', 'training', 'talent', 'compensation', 'compliance', 'workforce']
    },
    'Healthcare': {
        'desc': 'patient care clinical medical nursing diagnosis treatment healthcare management hospital pharmacy surgery medical records health insurance anatomy physiology patient safety',
        'skills': ['patient care', 'clinical', 'medical', 'nursing', 'pharmacy', 'diagnosis', 'treatment', 'hospital']
    },
    'Engineering': {
        'desc': 'mechanical electrical civil structural design autocad project management manufacturing quality control thermodynamics materials testing prototyping technical drawing specifications',
        'skills': ['autocad', 'design', 'manufacturing', 'quality', 'mechanical', 'electrical', 'project management', 'testing']
    },
}

CHART = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#ffffff',
    font=dict(color='#a8a29e', family='DM Sans', size=12),
    margin=dict(l=0, r=0, t=10, b=0)
)
GRID = dict(gridcolor='#f5f3ef', zeroline=False, showline=True, linecolor='#e7e5e4')


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    return ' '.join(words)


def get_grade(score):
    if score >= 60: return 'Excellent', 'excellent'
    elif score >= 40: return 'Good', 'good'
    else: return 'Average', 'average'


def get_skill_gap(resume_text, role):
    skills = JOB_DESCRIPTIONS[role]['skills']
    resume_lower = resume_text.lower()
    present = [s for s in skills if s in resume_lower]
    missing = [s for s in skills if s not in resume_lower]
    return present, missing


@st.cache_resource
def load_data():
    df = pd.read_csv('Resume.csv')
    df = df.dropna(subset=['Resume_str', 'Category'])
    df['clean'] = df['Resume_str'].apply(clean_text)
    return df


@st.cache_resource
def build_vectorizer(texts):
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    vec.fit(texts)
    return vec


def screen_resumes(df, vec, role, top_n=8):
    job_desc = clean_text(JOB_DESCRIPTIONS[role]['desc'])
    role_key = role.upper().replace(' ', '-')
    if role_key in df['Category'].values:
        pool = df[df['Category'] == role_key].copy()
    else:
        pool = df.copy()
    pool = pool.reset_index(drop=True)

    all_texts = [job_desc] + list(pool['clean'])
    tfidf = vec.transform(all_texts)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:])[0]

    pool['score'] = (sims * 100).round(1)
    ranked = pool.nlargest(top_n, 'score').reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)
    return ranked


with st.spinner("Loading 2,484 resumes..."):
    df = load_data()
    all_texts = list(df['clean'])
    vec = build_vectorizer(all_texts)

cat_counts = df['Category'].value_counts()


with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 18px'>
        <div style='font-family:Playfair Display,serif;font-size:18px;color:#dc2626'>Resume Screener</div>
        <div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#d97706;margin-top:4px'>Candidate Ranking System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#d97706;margin-bottom:8px'>Select a job role</div>", unsafe_allow_html=True)
    selected_role = st.selectbox("Role", list(JOB_DESCRIPTIONS.keys()), label_visibility="collapsed")
    screen_btn = st.button("Screen resumes for this role")

    st.markdown("---")
    st.markdown("<div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#d97706;margin-bottom:10px'>Page guide</div>", unsafe_allow_html=True)
    sections = [
        ("4 key numbers", "Dataset snapshot"),
        ("Top candidates", "Ranked by match score"),
        ("Skill gap", "What each candidate is missing"),
        ("Score distribution", "How candidates spread out"),
        ("Category chart", "Resume breakdown by field"),
        ("How scoring works", "Logic explained"),
    ]
    for title, desc in sections:
        st.markdown(f"""
        <div style='border-left:2px solid #d97706;padding:5px 10px;margin-bottom:7px'>
            <div style='font-size:12px;color:#dc2626;font-weight:1000'>{title}</div>
            <div style='font-size:11px;color:#d97706;margin-top:1px;line-height:1.5'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px;color:#d97706;line-height:2.2'>
        Resumes — 2,484<br>
        Job categories — 24<br>
        Method — TF-IDF + Cosine similarity<br>
        Source — Kaggle
    </div>
    """, unsafe_allow_html=True)


results = None
if screen_btn:
    with st.spinner(f"Screening resumes for {selected_role}..."):
        results = screen_resumes(df, vec, selected_role, top_n=8)


st.markdown('<div class="headline">Who is the right fit for this role?</div>', unsafe_allow_html=True)
st.markdown('<div class="subline"><strong>An ML system that reads 2,484 real resumes, scores each one against a job description and ranks candidates by how well they match — with a full skill gap breakdown for each.</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hint-box">
    Select a job role from the left sidebar and hit screen. The system scores every resume against
    that role's requirements using TF-IDF similarity, ranks the top candidates and shows exactly
    which skills each one has and which ones they are missing.
</div>
""", unsafe_allow_html=True)


s1, s2, s3, s4 = st.columns(4, gap="small")
stats = [
    ("#d97706", "Total resumes", "2,484", "Real resumes across 24 job categories."),
    ("#16a34a", "Job roles available", str(len(JOB_DESCRIPTIONS)), "Select any role to screen against."),
    ("#dc2626", "Scoring method", "TF-IDF", "Measures how relevant each resume is to the job description."),
    ("#1c1917", "Top candidates shown", "8", "The highest scoring resumes for your selected role."),
]
for col, (color, label, num, desc) in zip([s1, s2, s3, s4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-top" style="background:{color}"></div>
            <div class="stat-label"><strong>{label}</strong></div>
            <div class="stat-num">{num}</div>
            <div class="stat-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if results is not None:
    st.markdown('<div class="section-label">Screening results</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-title">Top candidates for {selected_role}</div>', unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9], gap="small")

    with left:
        for _, row in results.iterrows():
            grade_label, grade_class = get_grade(row['score'])
            present, missing = get_skill_gap(row['Resume_str'], selected_role)
            bar_color = '#16a34a' if grade_class == 'excellent' else '#d97706' if grade_class == 'good' else '#dc2626'

            present_html = ''.join([f'<span class="skill-present">{s}</span>' for s in present]) if present else '<span style="font-size:12px;color:#a8a29e">None detected</span>'
            missing_html = ''.join([f'<span class="skill-missing">{s}</span>' for s in missing]) if missing else '<span style="font-size:12px;color:#16a34a">No gaps found</span>'

            st.markdown(f"""
            <div class="candidate-card {grade_class}">
                <div class="rank-num">Rank {row['rank']}</div>
                <div style='display:flex;align-items:center;gap:0;margin-bottom:6px'>
                    <span class="cand-score">{row['score']:.1f}%</span>
                    <span class="cand-grade grade-{grade_class}">{grade_label}</span>
                </div>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:10px'>
                    <div class="score-bar-wrap">
                        <div class="score-bar" style="width:{row['score']}%;background:{bar_color}"></div>
                    </div>
                    <span style='font-size:11px;color:#a8a29e'>{row['score']:.1f}%</span>
                </div>
                <div style='margin-bottom:6px'>
                    <div style='font-size:11px;color:#a8a29e;margin-bottom:4px'>Skills present</div>
                    {present_html}
                </div>
                <div>
                    <div style='font-size:11px;color:#a8a29e;margin-bottom:4px'>Skills missing</div>
                    {missing_html}
                </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-label">Score distribution</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">How candidates spread out</div>', unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px;color:#a8a29e;margin-bottom:16px;font-weight:300'>Match scores for all top candidates. Green means strong fit, amber is decent, red needs development.</div>", unsafe_allow_html=True)

        score_colors = []
        for s in results['score']:
            if s >= 60: score_colors.append('#16a34a')
            elif s >= 40: score_colors.append('#d97706')
            else: score_colors.append('#dc2626')

        fig = go.Figure(go.Bar(
            x=[f"#{r}" for r in results['rank']],
            y=results['score'],
            marker=dict(color=score_colors),
            hovertemplate='Rank %{x} — Score: %{y:.1f}%<extra></extra>'
        ))
        fig.add_hline(y=60, line_dash='dot', line_color='#16a34a',
                      annotation_text='Excellent threshold',
                      annotation_font_color='#16a34a')
        fig.add_hline(y=40, line_dash='dot', line_color='#d97706',
                      annotation_text='Good threshold',
                      annotation_font_color='#d97706')
        fig.update_layout(
            **CHART,
            xaxis=dict(**GRID),
            yaxis=dict(**GRID, range=[0, 100], ticksuffix='%'),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div style='font-size:13px;font-weight:600;color:#1c1917;margin:20px 0 10px'>Grade breakdown</div>", unsafe_allow_html=True)
        grades = results['score'].apply(lambda s: get_grade(s)[0])
        grade_counts = grades.value_counts()
        g_colors = {'Excellent': '#16a34a', 'Good': '#d97706', 'Average': '#dc2626'}
        fig2 = go.Figure(go.Bar(
            x=list(grade_counts.index),
            y=list(grade_counts.values),
            marker=dict(color=[g_colors.get(g, '#a8a29e') for g in grade_counts.index]),
            hovertemplate='%{x} — %{y} candidates<extra></extra>'
        ))
        fig2.update_layout(
            **CHART,
            xaxis=dict(**GRID),
            yaxis=dict(**GRID, tickformat='d'),
            height=200
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("""
    <div style='background:#fff;border-radius:8px;padding:48px;text-align:center;border:1px solid #e7e5e4'>
        <div style='font-size:16px;color:#a8a29e;margin-bottom:8px'>No role selected yet</div>
        <div style='font-size:13px;color:#d6d3d1;line-height:1.7'>
            Pick a job role from the left sidebar and hit screen to see the top candidates ranked by match score.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="section-label">Resume pool</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">All 2,484 resumes by category</div>', unsafe_allow_html=True)
st.markdown("<div style='font-size:13px;color:#a8a29e;margin-bottom:16px;font-weight:500'>The full dataset broken down by job category. The model screens across all of these when you select a role.</div>", unsafe_allow_html=True)

sorted_cats = cat_counts.sort_values(ascending=True).tail(16)
fig3 = go.Figure(go.Bar(
    x=sorted_cats.values, y=sorted_cats.index,
    orientation='h',
    marker=dict(
        color=list(range(len(sorted_cats))),
        colorscale=[[0, '#e7e5e4'], [0.5, '#f59e0b'], [1, '#d97706']],
        showscale=False
    ),
    hovertemplate='%{y} — %{x} resumes<extra></extra>'
))
fig3.update_layout(
    **CHART,
    xaxis=dict(**GRID, tickformat='d', tickfont=dict(color='#dc2626')),
    yaxis=dict(gridcolor='rgba(0,0,0,0)', zeroline=False, tickfont=dict(color='#dc2626')),
    height=440
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">The logic behind the scores</div>', unsafe_allow_html=True)

a1, a2, a3 = st.columns(3, gap="small")
actions = [
    ("Step 01", "Resume text is cleaned",
     "Each resume goes through text cleaning — links, punctuation and common words are removed. What remains is the <span class='ac-hi'>meaningful skill and experience vocabulary</span> of that candidate."),
    ("Step 02", "TF-IDF scores relevance",
     "Every word in the resume gets a weight based on how <span class='ac-hi'>important it is to the job description</span>. Words that appear often in the job desc but rarely elsewhere score higher."),
    ("Step 03", "Cosine similarity ranks candidates",
     "The resume vector is compared to the job description vector. The closer they are, the higher the score. <span class='ac-hi'>100% would be a perfect match</span> — in practice, 60%+ is considered excellent."),
]
for col, (num, title, body) in zip([a1, a2, a3], actions):
    with col:
        st.markdown(f"""
        <div class="action-card">
            <div class="ac-num" style="color:#dc2626;font-weight:700;">{num}</div>
            <div class="ac-title">{title}</div>
            <div class="ac-body">{body}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#a8a29e;font-size:11px;letter-spacing:1.5px;
            text-transform:uppercase;border-top:1px solid #e7e5e4;
            padding-top:20px;margin-top:40px'>
    Future Interns ML Internship &nbsp;·&nbsp;
    Resume Dataset, Kaggle &nbsp;·&nbsp; TF-IDF + Cosine Similarity
</div>
""", unsafe_allow_html=True)