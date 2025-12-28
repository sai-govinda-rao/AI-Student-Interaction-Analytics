import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Student Interaction Analytics",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# Load Data & Models
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("../Data/ai_assistant_usage_student_life.csv")

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("../Models/logistic_regression.pkl"),
        "Random Forest": joblib.load("../Models/random_forest.pkl"),
        "Naive Bayes": joblib.load("../Models/naive_bayes.pkl"),
        "Decision Tree": joblib.load("../Models/decision_tree.pkl")
    }
    return models

models = load_models()
df = load_data()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("Menu")
menu = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Overview",
        "📊 Exploratory Data Analysis",
        "📈 Behavioral Insights",
        "🤖 ML Predictions",
        "👨‍💻 About Project"
    ]
)

# ==================================================
# HOME
# ==================================================
if menu == "🏠 Overview":
    st.title("🎓 AI Assistant Usage Analytics & Student Retention Prediction")

    st.markdown("""
    **An end-to-end Data Science project that analyzes how students interact with AI assistants 
    and predicts whether they will reuse AI tools based on behavior, task type, and satisfaction.**
    """)

    st.markdown("""
    ### Problem Statement
    With AI tools like ChatGPT becoming common in education, institutions and EdTech platforms 
    need to understand **how students use AI**, **what drives satisfaction**, and **what factors influence continued usage**.

    However, real-world datasets for this problem are scarce due to privacy concerns.
    This project addresses that gap using a **realistic synthetic dataset** and applies
    data science techniques to extract actionable insights.
    """)


    st.markdown("""
    **Synthetic dataset of 10,000 student–AI interaction sessions**  
    designed to analyze how learners engage with AI tools like ChatGPT, GeminiAI, GrokAI, ...etc.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Sessions", len(df))
    col2.metric("Disciplines", df["Discipline"].nunique())
    col3.metric("Avg Session Time", f"{df['SessionLengthMin'].mean():.1f} min")
    col4.metric("Reuse Rate", f"{df['UsedAgain'].mean()*100:.1f}%")

    st.markdown("""
    ### Practical Use Cases
    - **EdTech platforms**: Improve AI assistant design and retention
    - **Educational analytics teams**: Understand student engagement patterns
    - **AI product teams**: Measure perceived usefulness and satisfaction
    - **Researchers & learners**: Practice EDA, feature engineering, and ML pipelines
    """)

    st.success("This project showcases the complete Data Science workflow — from raw data to deployed ML predictions.")

# ==================================================
# EDA
# ==================================================
elif menu == "📊 Exploratory Data Analysis":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.markdown("""
    This section explores how students interact with AI assistants across 
    different academic levels, tasks, and satisfaction levels.
    Each visualization is followed by a short, clear insight.
    """)

    st.divider()

    # --------------------------------------------------
    # 1. Session Length Distribution
    # --------------------------------------------------
    st.subheader("Distribution of AI Session Length")

    fig, ax = plt.subplots()
    sns.histplot(df["SessionLengthMin"], bins=30, kde=True, ax=ax)
    ax.set_xlabel("Session Length (minutes)")
    ax.set_ylabel("Number of Sessions")
    st.pyplot(fig)

    st.info("""
    **Insight:**  
    Most AI interactions are relatively short, indicating that students 
    primarily use AI assistants for **quick problem-solving or clarification** 
    rather than long continuous sessions.
    """)

    st.divider()

    # --------------------------------------------------
    # 2. Student Level Distribution
    # --------------------------------------------------
    st.subheader("Student Level Distribution")

    fig, ax = plt.subplots()
    df["StudentLevel"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

    st.info("""
    **Insight:**  
    Undergraduate students form the largest user group, suggesting that 
    AI tools are most actively used during **core academic and skill-building years**.
    """)

    st.divider()

    # --------------------------------------------------
    # 3. Task Type Frequency
    # --------------------------------------------------
    st.subheader("AI Usage by Task Type")

    fig, ax = plt.subplots()
    sns.countplot(
        y="TaskType",
        data=df,
        order=df["TaskType"].value_counts().index,
        ax=ax
    )
    ax.set_xlabel("Number of Sessions")
    ax.set_ylabel("Task Type")
    st.pyplot(fig)

    st.info("""
    **Insight:**  
    Coding and writing-related tasks dominate AI usage, highlighting that 
    students rely on AI most for **technical assistance and content generation**.
    """)

    st.divider()

    # --------------------------------------------------
    # EDA Summary
    # --------------------------------------------------
    st.success("""
    **EDA Summary:**  
    - Students primarily use AI for short, task-focused interactions  
    - Coding and writing are the most common AI-supported activities  
               
    These insights directly motivate the machine learning models 
    used later to predict **AI reuse behavior**.
    """)


# ==================================================
# INSIGHTS
# ==================================================
elif menu == "📈 Behavioral Insights":
    st.title("📈 Key Behavioral Insights")
    
    # --------------------------------------------------
    # 4. AI Assistance Level vs Satisfaction
    # --------------------------------------------------
    st.subheader(" AI Assistance Level vs Satisfaction")

    fig, ax = plt.subplots()
    sns.lineplot(
        x="AI_AssistanceLevel",
        y="SatisfactionRating",
        data=df,
        marker="o",
        ax=ax
    )
    ax.set_xlabel("AI Assistance Level")
    ax.set_ylabel("Average Satisfaction Rating")
    st.pyplot(fig)

    st.info("""
    **Insight:**  
    Satisfaction increases consistently as perceived AI assistance improves, 
    confirming that **usefulness is a key driver of positive user experience**.
    """)

    st.divider()

    # --------------------------------------------------
    # 5. Task Type vs Satisfaction
    # --------------------------------------------------
    st.subheader(" Satisfaction Across Different Task Types")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        x="TaskType",
        y="SatisfactionRating",
        data=df,
        ax=ax
    )
    ax.set_xlabel("Task Type")
    ax.set_ylabel("Satisfaction Rating")
    st.pyplot(fig)

    st.info("""
    **Insight:**  
    Tasks such as coding and research tend to have higher satisfaction scores,
    suggesting that AI provides **more consistent value in structured problem domains**.
    """)

    st.divider()

    st.subheader(" AI Assistance Level vs Average Session Duration")

    fig, ax = plt.subplots()
    sns.lineplot(
        x="AI_AssistanceLevel",
        y="SessionLengthMin",
        data=df,
        estimator="mean",
        marker="o",
        ax=ax
    )

    ax.set_xlabel("AI Assistance Level")
    ax.set_ylabel("Average Session Length (minutes)")
    st.pyplot(fig)

    st.info("""
    **Behavioral Insight:**  
    As AI assistance improves, session length stabilizes, suggesting that effective AI 
    helps students **complete tasks efficiently rather than prolonging interactions**.
    """)


    st.divider()

    st.subheader(" AI Reuse Behavior Across Task Types")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(
        x="TaskType",
        y="UsedAgain",
        data=df,
        ax=ax
    )

    ax.set_xlabel("Task Type")
    ax.set_ylabel("Probability of Reuse")
    st.pyplot(fig)

    st.info("""
    **Behavioral Insight:**  
    Students working on coding and research tasks show higher reuse rates,
    indicating that AI tools are perceived as **more valuable for complex, 
    problem-solving-oriented activities**.
    """)

    st.success("""
    **Behavioral Summary:**  
    - User satisfaction is the strongest predictor of AI reuse  
    - Effective AI assistance improves efficiency, not dependency  
    - Task context plays a significant role in repeat engagement  

    These insights guided the design of the machine learning models 
    used to predict **AI tool reuse behavior**.
    """)



# ==================================================
# ML PREDICTIONS
# ==================================================
elif menu == "🤖 ML Predictions":
    st.title("🤖 AI Reuse Prediction Using Multiple ML Models")


    st.markdown("""
    This section allows you to compare multiple machine learning models
    and predict whether a student is likely to reuse an AI assistant.
    """)

    student_level_map = {
        "High School": 0,
        "Undergraduate": 1,
        "Graduate": 2
    }

    # ------------------------------
    # Model Selection
    # ------------------------------
    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        list(models.keys())
    )


    selected_model = models[selected_model_name]

    st.divider()

    # ------------------------------
    # User Input Section
    # ------------------------------
    st.markdown("### Enter Student Session Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        student_level_label = st.selectbox(
            "Student Level",
            ["High School", "Undergraduate", "Graduate"]
        )
        discipline = st.selectbox("Discipline", sorted(df["Discipline"].unique()))
        finalOutcome = st.selectbox("FinalOutcome", ['Assignment Completed', 'Idea Drafted', 'Confused', 'Gave Up'])

    with col2:
        task_type = st.selectbox("Task Type", sorted(df["TaskType"].unique()))
        session_length = st.slider("Session Length (min)", 1, 120, 30)
        satisfaction = st.slider("Satisfaction Rating", 1, 5, 3)

    with col3:
        prompts = st.slider("Total Prompts", 1, 50, 10)
        ai_help = st.slider("AI Assistance Level", 1, 5, 3)

    # Convert Student Level for model
    student_level = student_level_map[student_level_label]

    st.divider()

    if st.button("Predict Outcome"):
        input_df = pd.DataFrame({
            "StudentLevel": [student_level],
            "Discipline": [discipline],
            "SessionLengthMin": [session_length],
            "TotalPrompts": [prompts],
            "TaskType": [task_type],
            "AI_AssistanceLevel": [ai_help],
            "FinalOutcome": [finalOutcome],
            "SatisfactionRating": [satisfaction]
        })

        prediction = selected_model.predict(input_df)[0]
        probability = selected_model.predict_proba(input_df)[0][1]

        st.success(
            f"Will the student reuse the AI? → **{'Yes' if prediction else 'No'}**"
        )
        st.info(
            f"Confidence Score: **{probability:.2%}**"
        )


# ==================================================
# ABOUT PROJECT
# ==================================================
elif menu == "👨‍💻 About Project":
    st.title("👨‍💻 About the Project")

    st.markdown("""
    ### 🎓 AI Assistant Usage Analytics & Retention Prediction

    This project analyzes **how students interact with AI assistants** 
    (such as ChatGPT-like tools) and predicts whether they are likely 
    to **reuse AI tools** based on behavioral and contextual factors.

    The dataset is **fully synthetic yet realistic**, designed to mirror 
    real-world student–AI interaction patterns while avoiding privacy 
    and ethical concerns.
    """)

    st.divider()

    # --------------------------------------------------
    # Problem & Motivation
    # --------------------------------------------------
    st.markdown("""
    ### Problem & Motivation
    As AI tools become increasingly integrated into education, institutions 
    and EdTech platforms need data-driven answers to key questions:

    - What types of students use AI tools the most?
    - Which tasks benefit most from AI assistance?
    - What factors influence satisfaction and continued AI usage?

    Due to the lack of publicly available datasets in this domain, 
    this project addresses the gap using a controlled synthetic dataset 
    and modern machine learning techniques.
    """)

    st.divider()

    # --------------------------------------------------
    # Approach & Methodology
    # --------------------------------------------------
    st.markdown("""
    ### Approach & Methodology
    - Performed **exploratory data analysis (EDA)** to uncover usage patterns
    - Engineered behavioral features related to engagement and task context
    - Built multiple **machine learning pipelines** using scikit-learn
    - Compared models including Logistic Regression, Random Forest, 
      Naive Bayes, and Decision Tree
    - Implemented **production-ready inference** using Streamlit
    """)

    st.divider()

    # --------------------------------------------------
    # Key Outcomes
    # --------------------------------------------------
    st.markdown("""
    ### Key Outcomes & Insights
    - Student satisfaction is the strongest driver of AI reuse
    - AI tools are most valuable for structured tasks like coding and research
    - Effective AI assistance improves efficiency rather than dependency
    - Behavioral features can reliably predict reuse behavior
    """)

    st.divider()

    # --------------------------------------------------
    # Skills Demonstrated
    # --------------------------------------------------
    st.markdown("""
    ### Skills Demonstrated
    - Python for data analysis and modeling  
    - Exploratory Data Analysis & data storytelling  
    - Feature engineering for behavioral data  
    - Machine Learning pipelines & model comparison  
    - Model deployment with Streamlit  
    """)

    st.divider()

    # --------------------------------------------------
    # Real-World Relevance
    # --------------------------------------------------
    st.markdown("""
    ### Real-World Relevance
    This project demonstrates how data science can be applied to:
    - Improve AI-powered educational products
    - Analyze user engagement and retention
    - Support data-driven product decisions in EdTech platforms
    """)

    st.success("""
    This project showcases my ability to take a data science problem 
    from raw data to a deployable, interactive machine learning application.
    """)

    # --------------------------------------------------
    # Links
    # --------------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.link_button("💼 LinkedIn", "https://www.linkedin.com/in/kornu-sai-govinda-rao-b077a9286/")

    with col2:
        st.link_button("📂 GitHub", "https://github.com/sai-govinda-rao")

