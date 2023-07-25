import os
import streamlit as st
from dotenv import load_dotenv
from elevenlabs import generate, set_api_key
from elevenlabs.api import Voices
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks.base import BaseCallbackHandler
from streamlit_player import st_player
# from audio_recorder_streamlit import audio_recorder
import re
import uuid
from langchain import PromptTemplate
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENVIRONMENT')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CREATIVITY = os.getenv('CREATIVITY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
STABILITY = os.getenv('STABILITY')
SIMILARITY_BOOST = os.getenv('SIMILARITY_BOOST')
FINE_TUNE_VOICES =  eval(str(os.getenv('FINE_TUNE_VOICES')))
ENABLE_ELEVENLABS = eval(str(os.getenv('ENABLE_ELEVENLABS')))
ELEVENLABS_CHARACTER_LIMIT = os.getenv('ELEVENLABS_CHARACTER_LIMIT')
FAQ1 = os.getenv("FAQ1")
FAQ2 = os.getenv("FAQ2")
FAQ3 = os.getenv("FAQ3")
FAQ4 = os.getenv("FAQ4")
FAQ5 = os.getenv("FAQ5")
PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE")
# st.markdown("# Hello World!")
# st.chat_input("Hello, I am a helpful assistant. It depends on you how helpful I am!")
# st.audio(data="https://github.com/newdevorder/The-Nathan-Latka-Chatbot/blob/main/whogaveyouthislink.wav", format="audio/wav")
st.set_page_config(page_title="Chat with Nathan Latka!", page_icon="assets/nathan.jpg")
title, brand_icon = st.columns([0.7, 0.3])
with title:
    st.title("Chat with Nathan from Founderpath.com!")
with brand_icon:
    st.image("assets/founderpath-lg.svg")

# @st.cache_resource
def configure_qa_chain():
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key
    )

    docs_db = Pinecone.from_existing_index(PINECONE_INDEX, embeddings)
    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Setup LLM and QA chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=CREATIVITY, streaming=True
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=docs_db.as_retriever(), memory=memory, verbose=True,
    )
    return qa_chain

openai_api_key = st.sidebar.text_input(key="openai_api_key", label="OpenAI API Key", type="password")
print(openai_api_key)
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Additional Context")

    # def on_retriever_start(self, query: str, **kwargs):
    #     # print(**kwargs)
    #     # print(query)
    #     self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        # print("Hi")
        # st.info(documents)
        reference = []
        for idx, doc in enumerate(documents):
            # st.info("https://youtu.be/"+idx)
            source = os.path.basename(doc.metadata["source"])
            if source not in reference:
                print(source)
                print("https://youtu.be/"+re.sub(r'\.txt$', '', source))
                reference.append(source)
                st_player(key=str(uuid.uuid4()), url="https://youtu.be/"+re.sub(r'\.txt$', '', source))
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)

# New Code
def generate_and_play(audio_text):
    print("Generating audio")
    set_api_key(ELEVENLABS_API_KEY)
    # Generate audio using ElevenLabs
    audio = generate(text=audio_text, voice=getVoice("Bella"), 
                     model="eleven_monolingual_v1")

    # Play the audio
    st.audio(audio, format="audio/wav", start_time=0, sample_rate=None)
# End of New Code

def getVoice(voice_name):
    # Get available voices from api.
    voices = Voices.from_api()
    found_voices = [voice for voice in voices if voice.name == voice_name]
    if len(found_voices) >= 1:
        found_voice=found_voices[0]
        if(FINE_TUNE_VOICES):
            found_voice.settings.stability = STABILITY
            found_voice.settings.similarity_boost = SIMILARITY_BOOST
        return found_voices[0]
    else:
        return voices[0]

def get_condensed_answer(response):
    if int(ELEVENLABS_CHARACTER_LIMIT) <= 0:
        return response
    else:
        condensed_answer = response[0:int(ELEVENLABS_CHARACTER_LIMIT)]
        print('The following characters will be converted to audio' + condensed_answer)
        return condensed_answer

qa_chain = configure_qa_chain()

# FAQs
st.sidebar.markdown("Here are some popular questions that people ask me.")
if FAQ1:
    st.sidebar.markdown("1. " + FAQ1)
if FAQ2:
    st.sidebar.markdown("2. " + FAQ2)
if FAQ3:
    st.sidebar.markdown("3. " + FAQ3)
if FAQ4:
    st.sidebar.markdown("4. " + FAQ4)
if FAQ5:
    st.sidebar.markdown("5. " + FAQ5)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")
# audio_bytes = audio_recorder(text='', icon_size="15")
# if audio_bytes:
#     st.audio(audio_bytes, format="audio/wav")


# template = """\
# You are a naming consultant for new companies.
# What is a good name for a company that makes {user_original_query}?
# """

# user_query = PromptTemplate.from_template(user_original_query)
# user_query.format(product="colorful socks")


# prompt = PromptTemplate(
#     template="You are a helpful assistant that translates {input_language} to {output_language}.",
#     input_variables=["input_language", "output_language"],
# )
if user_query:
    # system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    # human_template = "{text}"
    # human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    # print(human_message_prompt)
    # chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    # chat_prompt.format_prompt(text=user_query)

    # print(chat_prompt)
    template = PROMPT_TEMPLATE + ": {text}"
    # template = "I want you to act like a helpful AI assistant who helps Software As A Service (SAAS) founders generate revenue and make stratigic decisions based on the context. You will suggest relevant metrics like CAC, LTV, Net Dollar Retention, and MRR as appropriate, and quote the relevant section in the context: {text}"
    prompt_template = PromptTemplate.from_template(template)
    prompt = prompt_template.format(text=user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())

        try:
            response = qa_chain.run(prompt, callbacks=[retrieval_handler])
            st.session_state.messages.append({"role": "assistant", "content": response})
        
            #Enable or Disable Eleven Labs
            if ENABLE_ELEVENLABS:
                print('Hello.. Eleven Labs!')
                condensed_answer = get_condensed_answer(response)
                # New Code
                # Use ElevenLabs API to generate speech and play it
                if ELEVENLABS_API_KEY:
                    generate_and_play(audio_text=condensed_answer)
                else:
                    st.error("ElevenLabs API key not found. Please set the 'ELEVENLABS_API_KEY' environment variable.")

            st.write(response)

            # Visit FounderPath: Where Bootstrapped SaaS Founders Get Capital
            # https://founderpath.com/
            st.markdown("Visit [FounderPath](https://founderpath.com/): Where Bootstrapped SaaS Founders Get Capital. We raised $145M to [fund bootstrappers](https://twitter.com/NathanLatka/status/1557026126639370241). Follow [Nathan Latka on twitter @NathanLatka](https://twitter.com/NathanLatka)")
        except Exception as ex:
            print(ex)
            st.write("An unexpected error has occured.")