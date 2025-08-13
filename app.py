import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from utils import SentimentAnalyzer, MoodTracker, get_quote
from dotenv import load_dotenv
import os

# if not st.secrets:
#     load_dotenv()

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY") # or st.secrets["GROQ_API_KEY"]

# Initialize Streamlit app
st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠")
st.title("🧠 Mental Health Support Chatbot")
st.write("I'm here to listen and support you. Share how you're feeling, and I'll provide insights and encouragement.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How are you feeling today?"}]
if "mood_tracker" not in st.session_state:
    st.session_state.mood_tracker = MoodTracker()
if "sentiment_analyzer" not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()

# Initialize LangChain with Groq
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)
memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Type your message here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Analyze sentiment
    sentiment, polarity = st.session_state.sentiment_analyzer.analyze_sentiment(prompt)
    st.session_state.mood_tracker.update_mood(sentiment, polarity)

    # Generate response
    with st.spinner("Thinking..."):
        response = conversation.predict(input=prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Display mood summary and quote
if st.button("Show Mood Summary"):
    mood, behavior = st.session_state.mood_tracker.summarize_mood()
    quote = get_quote(mood)
    st.write("### Mood Summary")
    st.write(f"**Current Mood**: {mood.capitalize()}")
    st.write(f"**Behavior Analysis**: {behavior}")
    st.write(f"**Motivational Quote**: {quote}")