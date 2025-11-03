import streamlit as st
from rag_assist import ask_rag

st.title("🎓 RMIT Student Support Chatbot")
st.write("Ask me anything about enrolment, timetable, or student services!")
# 🪶 Custom styling
st.markdown(
    """
    <style>
    body { background-color: #0E1117; color: white; }
    .stTextInput>div>div>input {
        background-color: #1E1E1E;
        color: white;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

if st.button("Clear Chat"):
    st.rerun()

query = st.text_input("Type your question here:")

if query:
    with st.spinner("Thinking..."):
        answer, sources = ask_rag(query)
        st.success(answer)
        st.caption("📚 Sources:")
        for s in sources:
            st.write("-", s)
