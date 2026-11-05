
import streamlit as st
import json

from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="MovieSage AI",
    page_icon="🎬",
    layout="wide"
)

# --------------------------------
# Custom CSS
# --------------------------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #111827, #312e81);
    color: white;
}

.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #c4b5fd;
}

.subtitle {
    text-align: center;
    color: #d1d5db;
    font-size: 17px;
    margin-bottom: 30px;
}

div.stButton > button {
    background-color: #7c3aed;
    color: white;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: bold;
}

div.stButton > button:hover {
    background-color: #6d28d9;
    color: white;
}

[data-testid="stTextArea"] textarea {
    background-color: #1f2937;
    color: white;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# Pydantic Schema
# --------------------------------

class Movie(BaseModel):

    title: str
    release_year: Optional[str] = None
    director: Optional[str] = None
    genre: List[str] = Field(default_factory=list)
    main_cast: List[str] = Field(default_factory=list)
    language: Optional[str] = None
    country: Optional[str] = None
    rating: Optional[str] = None
    runtime: Optional[str] = None
    plot_overview: str
    budget: Optional[str] = None
    box_office_collection: Optional[str] = None


# --------------------------------
# Gemini Model
# --------------------------------

@st.cache_resource
def load_model():

    return ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0
    )


model = load_model()


# --------------------------------
# Output Parser
# --------------------------------

parser = PydanticOutputParser(
    pydantic_object=Movie
)


# --------------------------------
# Prompt Template
# --------------------------------

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are MovieSage, an AI assistant specialized
        in analyzing movies.

        Your task is to take raw, unstructured information
        about a movie and transform it into structured JSON.

        Follow these rules:

        1. Extract information only from the provided text.
        2. Do not invent information.
        3. If information is missing, use null or empty lists.
        4. Generate a clean and concise plot overview.
        5. Follow these formatting instructions:

        {format_instructions}
        """
    ),

    (
        "human",
        """
        Analyze the following raw movie information:

        {paragraph}
        """
    )

])


# --------------------------------
# LangChain Chain
# --------------------------------

chain = prompt | model | parser


# --------------------------------
# Main UI
# --------------------------------

st.markdown(
    '<div class="title">🎬 MovieSage AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Transform movie descriptions into structured information using AI</div>',
    unsafe_allow_html=True
)

st.divider()

# --------------------------------
# User Input
# --------------------------------

st.subheader("🎥 Enter Movie Information")

para = st.text_area(
    "Paste your movie paragraph here:",
    placeholder="Example: 3 Idiots is a Bollywood comedy-drama movie...",
    height=200
)

analyze = st.button(
    "✨ Analyze Movie",
    use_container_width=True
)


# --------------------------------
# Generate Output
# --------------------------------

if analyze:

    if not para.strip():

        st.warning("Please enter a movie paragraph first.")

    else:

        with st.spinner("🎬 MovieSage is analyzing your movie..."):

            try:

                movie = chain.invoke({

                    "paragraph": para,

                    "format_instructions":
                    parser.get_format_instructions()

                })

                st.success("Movie analysis completed!")

                # Store result
                st.session_state.movie_result = movie.model_dump()

            except Exception as e:

                st.error(f"Something went wrong: {e}")


# --------------------------------
# Display Result
# --------------------------------

if "movie_result" in st.session_state:

    movie = st.session_state.movie_result

    st.divider()

    st.header("📽️ Movie Analysis")

    # Movie title
    st.subheader(f"🎬 {movie['title']}")

    # Movie information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Release Year",
            movie["release_year"] or "Not available"
        )

    with col2:
        st.metric(
            "Runtime",
            movie["runtime"] or "Not available"
        )

    with col3:
        st.metric(
            "Rating",
            movie["rating"] or "Not available"
        )

    # Movie details
    st.subheader("📋 Movie Details")

    col1, col2 = st.columns(2)

    with col1:

        st.write("**Director:**", movie["director"] or "Not available")

        st.write("**Language:**", movie["language"] or "Not available")

        st.write("**Country:**", movie["country"] or "Not available")

        st.write("**Genre:**", ", ".join(movie["genre"]) or "Not available")

    with col2:

        st.write("**Main Cast:**", ", ".join(movie["main_cast"]) or "Not available")

        st.write("**Budget:**", movie["budget"] or "Not available")

        st.write(
            "**Box Office:**",
            movie["box_office_collection"] or "Not available"
        )

    # Plot Overview
    st.subheader("📖 Plot Overview")

    st.info(movie["plot_overview"])

    # JSON Output
    st.subheader("🔍 Structured JSON Output")

    json_output = json.dumps(
        movie,
        indent=4,
        ensure_ascii=False
    )

    st.code(json_output, language="json")

    # Download JSON
    st.download_button(
        label="⬇️ Download JSON",
        data=json_output,
        file_name="movie_analysis.json",
        mime="application/json",
        use_container_width=True
    )
