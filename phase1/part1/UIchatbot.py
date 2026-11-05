
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

load_dotenv()

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="Personality AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #172554);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
    color: #a5b4fc;
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 16px;
    margin-bottom: 30px;
}

[data-testid="stChatMessage"] {
    background-color: rgba(255,255,255,0.07);
    border-radius: 15px;
    padding: 15px;
}

.stButton button {
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Gemini Model
# -------------------------------

@st.cache_resource
def load_model():
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.8
    )

model = load_model()

# -------------------------------
# Personalities
# -------------------------------

personalities = {
    "Funny 😂": "You are a funny chatbot who responds with humor.",
    "Angry 😡": "You are an angry chatbot who responds in an irritated tone.",
    "Sad 😢": "You are a sad chatbot who responds in a melancholic tone.",
    "Romantic ❤️": "You are a romantic chatbot who responds in a romantic tone."
}

# -------------------------------
# Session State
# -------------------------------

if "personality" not in st.session_state:
    st.session_state.personality = "Funny 😂"

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(
            content=personalities[st.session_state.personality]
        )
    ]

# -------------------------------
# Sidebar
# -------------------------------

with st.sidebar:

    st.title("⚙️ Settings")

    selected_personality = st.selectbox(
        "Choose Your Personality",
        list(personalities.keys()),
        index=list(personalities.keys()).index(
            st.session_state.personality
        )
    )

    if selected_personality != st.session_state.personality:

        st.session_state.personality = selected_personality

        st.session_state.messages = [
            SystemMessage(
                content=personalities[selected_personality]
            )
        ]

        st.rerun()

    st.divider()

    st.write("### About")

    st.write(
        "An AI chatbot powered by Google Gemini "
        "and LangChain."
    )

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = [
            SystemMessage(
                content=personalities[
                    st.session_state.personality
                ]
            )
        ]

        st.rerun()

# -------------------------------
# Main UI
# -------------------------------

st.markdown(
    '<div class="main-title">🤖 Personality AI Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Chat with AI in your favorite personality style</div>',
    unsafe_allow_html=True
)

st.info(
    f"Current Personality: {st.session_state.personality}"
)

# -------------------------------
# Display Chat History
# -------------------------------

for message in st.session_state.messages[1:]:

    if isinstance(message, HumanMessage):

        with st.chat_message("user"):
            st.markdown(message.content)

    elif isinstance(message, AIMessage):

        with st.chat_message("assistant"):
            st.markdown(message.content)

# -------------------------------
# Chat Input
# -------------------------------

prompt = st.chat_input("Type your message here...")

if prompt:

    if prompt.strip() == "":

        st.warning("Please enter a message.")

    else:

        # Add user message
        user_message = HumanMessage(content=prompt)

        st.session_state.messages.append(user_message)

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    response = model.invoke(
                        st.session_state.messages
                    )

                    content = response.content

                    # Handle string or block-based content
                    if isinstance(content, list):
                        answer = "\n".join(
                            block.get("text", "")
                            if isinstance(block, dict)
                            else str(block)
                            for block in content
                        )
                    else:
                        answer = str(content)

                    st.markdown(answer)

                    # Store AI response
                    st.session_state.messages.append(
                        AIMessage(content=answer)
                    )

                except Exception as e:

                    st.error(f"Something went wrong: {e}")
