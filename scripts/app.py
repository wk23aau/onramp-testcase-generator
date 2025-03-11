# File: app.py
import streamlit as st
from generate_test_case import TestCaseGenerator

generator = TestCaseGenerator()

st.title("OnRamp Test Case Generator")
user_story_desc = st.text_area("User Story Description")
acceptance_criteria = st.text_area("Acceptance Criteria")

if st.button("Generate Test Case"):
    user_story = {
        "UserStoryDescription": user_story_desc,
        "AcceptanceCriteria": acceptance_criteria
    }
    test_case = generator.generate(user_story)
    st.write("Generated Test Case:")
    st.text(test_case)