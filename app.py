# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import time
import random
import numpy as np

# Paths (adjusted as needed)
THEORY_CSV = "math_theory.csv"
QUIZ_CSV = "quiz_bank.csv"
PROCESSED_SAMPLE = "processed_sample.csv"  # contains content_avg_correct and user_avg_correct
MODEL_PATH = "model/model.joblib"
ENC_PATH = "model/encoders.joblib"

@st.cache_data
def load_theory():
    return pd.read_csv(THEORY_CSV)

@st.cache_data
def load_quiz_bank():
    return pd.read_csv(QUIZ_CSV)

@st.cache_data
def load_processed_sample():
    try:
        return pd.read_csv(PROCESSED_SAMPLE)
    except:
        return pd.DataFrame()

@st.cache_resource
def load_model_and_encoders():
    model = joblib.load(MODEL_PATH)
    enc = joblib.load(ENC_PATH)
    return model, enc

theory_df = load_theory()
quiz_df = load_quiz_bank()
processed = load_processed_sample()
model, enc = load_model_and_encoders()

# --- Session state: initialize quiz/navigation keys safely ---
init_keys = {
    "mode": "Home",
    "mode_changed": False,
    "active_topic": None,
    "sampled": None,
    "current_q": 0,
    "answers": None,
    "start_times": None,
    "end_times": None,
    "quiz_started": False,
    "quiz_finished": False,
    "quiz_results": None,
    "quiz_seed": None,
}

for k, v in init_keys.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go_to_mode(new_mode: str):
    """Helper to switch app mode easily."""
    st.session_state["mode"] = new_mode

# YouTube links mapping
youtube_links = {
    "Algebra": ["https://www.youtube.com/watch?v=9Ib6eB90n8o&list=PLjm_mvBNlvBZ8d1EGhXxiVnviM4yPf-Uq"],
    "Geometry": ["https://www.youtube.com/watch?v=Gh0qoV9TdBI&list=PLjm_mvBNlvBbt8uSpdGmgYY9iiAMt_K_r"],
    "Trigonometry": ["https://www.youtube.com/watch?v=5IIYNc4-qZw"],
    "Calculus": ["https://www.youtube.com/watch?v=G-ti56DEXE8"],
    "Probability": ["https://www.youtube.com/watch?v=AfBPfqSoZ0Y"],
    "Statistics": ["https://www.youtube.com/watch?v=fFncQze5dBc"],
    "Number Theory": ["https://www.youtube.com/watch?v=5hD0aS0s4zM"],
    "Linear Algebra": ["https://www.youtube.com/watch?v=1XlT3Y2oyAU&list=PLU6SqdYcYsfI7Ebw_j-Vy8YKHdbHKP9am"],
    "Differential Equations": ["https://www.youtube.com/watch?v=bjJZKTrCBNw&list=PLU6SqdYcYsfIuZVt20v-eNZBfFLENrM1F"],
    "Set Theory": ["https://www.youtube.com/watch?v=5ZhNmKb-dqk"]
}
# Map part numbers to topic names
part_to_topic = {
    1: "Algebra",
    2: "Geometry",
    3: "Trigonometry",
    4: "Calculus",
    5: "Probability",
    6: "Statistics",
    7: "Number Theory",
    8: "Linear Algebra",
    9: "Differential Equations",
    10: "Set Theory"
}

# reverse mapping if needed later:
topic_to_part = {v: k for k, v in part_to_topic.items()}

st.sidebar.title("Learning Assistant")

if "mode" not in st.session_state:
    st.session_state["mode"] = "Home"

mode = st.sidebar.radio(
    "Select Mode",
    ["Home", "Theory", "Quiz", "Feedback", "Doubt Clarification"],
    index=["Home", "Theory", "Quiz", "Feedback","Doubt Clarification"].index(st.session_state["mode"]),
)
st.session_state["mode"] = mode

# --- HOME PAGE ---
if mode == "Home":
    st.title(" AI in Personalized Learning")
    st.markdown("""
    Welcome to your personalized learning platform!  
    Here you can **practice, revise, and get feedback** powered by AI.
    """)

    # Collect user info
    with st.form("user_info_form"):
        st.subheader("User Information")
        user_id = st.text_input("Enter your User ID")
        subject = st.selectbox("Select Subject", ["Mathematics", "Physics", "Chemistry"])
        topics = list(theory_df['topic'].unique())
        topic = st.selectbox("Topic", topics)

        submitted = st.form_submit_button("Save & Continue")

    # Save info in session state
    if submitted:
        if user_id and subject and topic:
            st.session_state["user_id"] = user_id
            st.session_state["subject"] = subject
            st.session_state["topic"] = topic
            st.success("Information saved! You can now explore Theory, Quiz, or Feedback.")
            st.markdown("<br>", unsafe_allow_html=True)
            st.button("-> Go to Theory", key="go_theory", on_click=lambda: go_to_mode("Theory"))
        else:
            st.warning("⚠ Please fill all details before proceeding.")

if "user_id" not in st.session_state or "topic" not in st.session_state:
    st.warning("⚠ Please go to the Home page and enter your details first.")
    st.stop()
user_id_input = st.session_state["user_id"]
topic = st.session_state["topic"]
subject = st.session_state["subject"]

if mode == "Doubt Clarification":
    st.subheader("Doubt Clarification")

    st.markdown("Ask any question related to your subject or topic — I’ll try to clarify it for you!")

    user_query = st.text_area("Type your doubt here:", height=150)

    if st.button("Get Clarification"):
        if user_query.strip():
            with st.spinner("Thinking..."):
                # Temporary simulation response
                st.write("**AI Tutor:**")
                st.success(f"This is a sample response. I'll explain '{user_query}' once the model is connected.")
        else:
            st.warning("Please enter a question first.")

if mode == "Theory":
    row = theory_df[theory_df['topic'] == topic]
    if not row.empty:
        content = row['theory'].values[0]
        # show markdown content
        st.markdown(content)
        # show youtube links if available
        if topic in youtube_links:
            st.markdown("**Recommended videos:**")
            for url in youtube_links[topic]:
                st.write(f"[Watch]({url})")
        st.markdown("<hr>", unsafe_allow_html=True)
        st.button("Attempt Quiz ->", key="go_quiz", on_click=lambda: go_to_mode("Quiz"))
    else:
        st.write("No theory available for this topic.")

if "q_start_times" not in st.session_state:
    st.session_state.q_start_times = {}

# --------- Quiz mode ----------
elif mode == "Quiz":
    st.subheader(f"Quiz: {topic}")

    topic_questions = quiz_df[quiz_df["topic"] == topic]
    if topic_questions.empty:
        st.warning("No quiz questions available for this topic.")
    else:
        # Initialize state on first load or when topic changes
        if "active_topic" not in st.session_state or st.session_state.active_topic != topic:
            st.session_state.active_topic = topic
            st.session_state.quiz_started = False
            st.session_state.quiz_finished = False

        # Only show number slider before quiz starts
        if not st.session_state.quiz_started and not st.session_state.quiz_finished:
            n_q = st.slider("Select number of questions", 3, 10, 5, key="num_qs")
            if st.button("Start Quiz"):
                sampled = topic_questions.sample(
                    n=min(n_q, len(topic_questions)),
                    random_state=random.randint(1, 10000)
                ).reset_index(drop=True)

                st.session_state.sampled = sampled
                st.session_state.current_q = 0
                st.session_state.answers = [None] * len(sampled)
                st.session_state.start_times = [time.time()] * len(sampled)
                st.session_state.end_times = [None] * len(sampled)
                st.session_state.quiz_started = True
                st.session_state.quiz_finished = False
                st.rerun()

        # Quiz in progress
        if st.session_state.get("quiz_started", False) and not st.session_state.get("quiz_finished", False):
            sampled = st.session_state.sampled
            current_q = st.session_state.current_q
            total_qs = len(sampled)
            question = sampled.iloc[current_q]

            st.markdown(f"### Q{current_q+1}/{total_qs}")
            st.markdown(question["question"])

            opt_text = str(question["options"])
            opts = [o.strip() for o in opt_text.split(",")] if "," in opt_text else ["Option 1", "Option 2"]

            selected = st.radio(
                "Select your answer:",
                opts,
                key=f"choice_{current_q}",
                index=opts.index(st.session_state.answers[current_q])
                if st.session_state.answers[current_q] in opts
                else None
            )
            st.session_state.answers[current_q] = selected

            # --- NAVIGATION ---
            col1, col2, col3 = st.columns([1, 6, 1])
            with col1:
                if st.button("<- Previous", disabled=current_q == 0):
                    st.session_state.end_times[current_q] = time.time()
                    st.session_state.current_q -= 1
                    st.rerun()

            with col3:
                if current_q < total_qs - 1:
                    if st.button("Next ->"):
                        st.session_state.end_times[current_q] = time.time()
                        st.session_state.current_q += 1
                        st.rerun()
                else:
                    if st.button("Submit Quiz"):
                        st.session_state.end_times[current_q] = time.time()

                        # --- Evaluate ---
                        sampled = st.session_state.sampled
                        details, correct_count, per_q_times = [], 0, []
                        for i, r in sampled.iterrows():
                            selected = st.session_state.answers[i]
                            start = st.session_state.start_times[i]
                            end = st.session_state.end_times[i] or time.time()
                            t_ms = max(1000, int((end - start) * 1000))
                            per_q_times.append(t_ms)

                            true_ans = str(r.get("answer", "")).strip().upper()
                            opts = [o.strip() for o in str(r["options"]).split(",")]
                            correct_text = (
                                opts[ord(true_ans) - ord("A")]
                                if true_ans and 0 <= (ord(true_ans) - ord("A")) < len(opts)
                                else true_ans
                            )
                            is_correct = (
                                selected and selected.strip().lower() == correct_text.lower()
                            )
                            if is_correct:
                                correct_count += 1

                            details.append(
                                {
                                    "question": r["question"],
                                    "selected": selected if selected else "Not answered",
                                    "true": correct_text,
                                    "is_correct": is_correct,
                                    "explanation": r.get("explanation", ""),
                                }
                            )

                        accuracy = correct_count / total_qs if total_qs else 0
                        avg_time_ms = np.mean(per_q_times)

                        st.session_state.quiz_results = {
                            "details": details,
                            "accuracy": accuracy,
                            "avg_time_ms": avg_time_ms,
                            "correct_count": correct_count,
                            "total_qs": total_qs,
                            "topic": topic,
                        }
                        st.session_state.quiz_finished = True
                        st.session_state.quiz_started = False
                        st.rerun()

        # Show results after submission
        if st.session_state.get("quiz_finished", False):
            res = st.session_state.quiz_results
            st.subheader("Quiz Summary")
            st.success(
                f"You answered {res['correct_count']}/{res['total_qs']} correctly — Accuracy: {res['accuracy']*100:.1f}%"
            )
            st.caption(f"Average time per question: {res['avg_time_ms']/1000:.1f} seconds")

            wrong_qs = [d for d in res["details"] if not d["is_correct"]]
            st.markdown("---")
            st.subheader("Review Incorrect Answers")

            if not wrong_qs:
                st.info("Perfect! All answers correct.")
            else:
                for i, d in enumerate(wrong_qs, start=1):
                    st.markdown(f"**Q{i}.** {d['question']}")
                    st.markdown(f"- ❌ Your answer: {d['selected']}")
                    st.markdown(f"- ✅ Correct: {d['true']}")
                    with st.expander(" Explanation"):
                        st.write(d["explanation"])
                    st.markdown("---")

            st.markdown("<hr>", unsafe_allow_html=True)
            colA, colB = st.columns(2)
            with colA:
                if st.button("See Feedback ->"):
                    go_to_mode("Feedback")
            with colB:
                if st.button("🔁 Attempt Again"):
                    # Reset everything
                    for k in [
                        "sampled",
                        "current_q",
                        "answers",
                        "start_times",
                        "end_times",
                        "quiz_finished",
                        "quiz_started",
                    ]:
                        if k in st.session_state:
                            del st.session_state[k]
                    st.rerun()


elif mode == "Feedback":
    st.subheader(" Personalized Feedback & Recommendations")

    # Ensure quiz results are available
    if "quiz_results" not in st.session_state:
        st.warning("⚠ Please complete a quiz first to view feedback.")
    else:
        # Retrieve stored results
        results = st.session_state["quiz_results"]
        accuracy = results["accuracy"]
        avg_time_ms = results["avg_time_ms"]
        total_qs = results["total_qs"]
        topic = results["topic"]

        st.markdown(f"### Topic: **{topic}**")
        st.write(f"Accuracy: **{accuracy*100:.1f}%**")
        st.write(f"Avg Time for Quiz: **{avg_time_ms/1000:.1f} s**")

        # Personalized Feedback
        st.markdown("---")
        st.subheader(" Personalized Feedback")
        feedback_msgs = []

        # Accuracy-based
        if accuracy < 0.4:
            feedback_msgs.append(" You seem to be struggling — revisit the theory and retry.")
        elif accuracy < 0.7:
            feedback_msgs.append(" Decent start — try more medium-level practice problems.")
        elif accuracy < 0.9:
            feedback_msgs.append(" Solid understanding — attempt some tougher questions.")
        else:
            feedback_msgs.append(" Excellent! You’ve mastered this topic — proceed to next chapter.")

        # Timing-based
        if avg_time_ms < 40000:
            feedback_msgs.append(" You're answering very quickly — slow down and check each step.")
        elif avg_time_ms > 100000:
            feedback_msgs.append(" You're taking quite long — practice more to build speed.")
        else:
            feedback_msgs.append(" Your pace is good — maintain this consistency!")

        for msg in feedback_msgs:
            st.write("- " + msg)

        # Model Prediction + Feature Importance
        if not processed.empty:
            try:
                part_val = topic_to_part.get(topic, 0)
                topic_avg_acc = processed[processed['part'] == part_val]['answered_correctly'].mean() if not processed.empty else 0.5

                feature_row = pd.DataFrame([{
                    "prior_question_elapsed_time": avg_time_ms,
                    "prior_question_had_explanation": 0,
                    "content_avg_correct": topic_avg_acc,
                    "user_avg_correct": accuracy,
                    "part_le": part_val,
                }])

                next_prob = model.predict_proba(feature_row)[0, 1]
                st.info(f" RandomForest predicted success probability for next topic: {next_prob*100:.1f}%")

            except Exception as e:
                st.warning(f"Prediction error: {e}")

            try:
                importances = pd.Series(model.feature_importances_, index=enc["feature_cols"])
                st.bar_chart(importances)
                st.caption("Feature Importance (Random Forest)")
            except Exception as e:
                st.warning(f"Could not show feature importances: {e}")

        # Personalized Topic Recommendation
        def pick_next_topic_for_user(user_id, epsilon=0.2):
            user_data = processed[processed['user_id'] == int(user_id)]
            if user_data.empty:
                return random.choice(processed['part'].dropna().unique().tolist())
            latest_performance = user_data.sort_values('content_id').groupby('part').tail(1)
            latest_performance['bandit_reward'] = 1 - latest_performance['user_part_accuracy_so_far']
            if random.random() < epsilon:
                return random.choice(latest_performance['part'].unique().tolist())
            best_row = latest_performance.sort_values('bandit_reward', ascending=False).iloc[0]
            return best_row['part']

        #  Recommendation
        st.markdown("---")
        st.subheader(" Personalized Recommendation")
        if accuracy >= 0.7:
            st.markdown(
                f"Great work! You scored *{accuracy*100:.1f}%*.\n\n"
                "You're ready to move on to the *next topic* "
            )
            if user_id_input:
                try:
                    recommended_part = pick_next_topic_for_user(user_id_input)
                    topic_name = part_to_topic.get(int(recommended_part), f"Part {recommended_part}")
                    st.subheader(f" Recommended Next Topic: {topic_name}")
                except Exception as e:
                    st.warning(f"Error picking topic: {e}")
        else:
            st.markdown(
                f" You scored *{accuracy*100:.1f}%*, consider revising this topic."
            )
            st.markdown("<hr>", unsafe_allow_html=True)
            st.button("🔁 Go Back to Theory", key="go_theory", on_click=lambda: go_to_mode("Theory"))
        st.markdown("<hr>", unsafe_allow_html=True)
        st.button("🔁 Go Back to Home", key="go_home", on_click=lambda: go_to_mode("Home"))
