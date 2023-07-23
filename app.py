from youtube_transcript_api import YouTubeTranscriptApi
import os
import multiprocessing
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from elevenlabs import generate, set_api_key
from elevenlabs.api import Voices
import scrapetube

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CREATIVITY = os.getenv('CREATIVITY')
ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
STABILITY = os.getenv('STABILITY')
SIMILARITY_BOOST = os.getenv('SIMILARITY_BOOST')
FINE_TUNE_VOICES =  eval(str(os.getenv('FINE_TUNE_VOICES')))
ENABLE_ELEVENLABS = eval(str(os.getenv('ENABLE_ELEVENLABS')))
# print("ENABLE_ELEVENLABS")
# print(ENABLE_ELEVENLABS)
# print(type(ENABLE_ELEVENLABS))
# print("FINE_TUNE_VOICES")
# print(type(FINE_TUNE_VOICES))
# print(FINE_TUNE_VOICES)

# Create a list to store the names of the generated .txt files
txt_files = []

# House Keeping
dir_path = "transcript_files"

# Initialization of youtube channel id session state
if 'youtube_channel_id' not in st.session_state:
    st.session_state["youtube_channel_id"] = ''

# Initialization of youtube video ids session state
if 'youtube_video_ids' not in st.session_state:
    st.session_state["youtube_video_ids"] = ['']

# Clear Dir
def clear_directory(dir_path):
  for file in os.scandir(dir_path):
    # print(file.path)
    os.remove(file.path)


def doc_preprocessing():
    loader = DirectoryLoader(
        dir_path,
        glob='**/*.txt',     # only the Txt files
        show_progress=False
    )
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    docs_split = text_splitter.split_documents(docs)
    return docs_split

# @st.cache_resource
def embedding_db():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    pinecone.describe_index(PINECONE_INDEX)
    
    docs_split = doc_preprocessing()
    print(docs_split)
    doc_db = Pinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name=PINECONE_INDEX
    )
    return doc_db

# @st.cache_resource
def configure_qa_chain():
    # Setup ChatOpenAI
    llm = ChatOpenAI(
        # openai_api_key=OPENAI_API_KEY,
        openai_api_key=openai_api_key,
        temperature=CREATIVITY,
        streaming=True
    )

    # Setup memory for contextual conversation
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Get Doc DB
    doc_db = embedding_db()

    # Configure QA Chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=doc_db.as_retriever(), memory=memory, verbose=True,
    )

    return qa_chain

# Function to Transcribe Youtube Video
def get_youtube_video_transcript(video_id):
  try:
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    # Combine all text from the transcript
    text = " ".join([x['text'] for x in transcript])
    # Use the video_id to generate file name
    file_name = f"{video_id}.txt"
    with open(dir_path+'/'+file_name, "w") as f:
        f.write(text)
    # Append the filename to the list
    txt_files.append(file_name)
    print(file_name + ' created.')
  except Exception as e:
    print(f"An error occurred for video_id: {video_id}. The error is {str(e)}.")

def transcribe_youtube_videos(videos_ids):
    i=0
    for video_id in videos_ids:
        i+=1
        print('Transcribing Video ID: ' + video_id)
        get_youtube_video_transcript(video_id)
        if i>=3:
            break

        # txt_files.append(video['videoId'])
        # print(video['videoId'])
        # print(video.properties)
    
    st.sidebar.info("Number of videos transcribed: "+str(i))

def initialize_folder(dir_path):
    if os.path.isdir(dir_path):
        print("Folder Exists.. Deleting the folder")
        clear_directory(dir_path)
        os.removedirs(dir_path)

    # Create the directory for new content
    os.makedirs(dir_path)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

youtube_channel_id = st.sidebar.text_input("Youtube Channel ID:")
youtube_video_ids = st.sidebar.text_input("Youtube Video IDs: Ex: r5UIEvoTtQ4, JiGvnl_kp-w, pCkUGVix5T4")

if not youtube_channel_id and not youtube_video_ids:
    print(youtube_channel_id)
    print(youtube_video_ids)
    st.info("Please add a youtube channel id or video ids to continue.")
    st.stop()
elif youtube_channel_id:
    print("Processing youtube channel id: " + youtube_channel_id)
    initialize_folder(dir_path)
    # Clear video ids session state
    youtube_video_ids = ''
    st.session_state["youtube_video_ids"] = ''
    print(st.session_state["youtube_channel_id"])
    # Delete directory
    if st.session_state["youtube_channel_id"] != youtube_channel_id:
        # Capture new youtube channel id
        st.session_state["youtube_channel_id"] = youtube_channel_id

        videos = scrapetube.get_channel(channel_username=youtube_channel_id) # NathanLatka "UC9-y-6csu5WGm29I7JiwpnA"
        print(videos)
        video_ids = [video['videoId'] for video in videos if video is not None]
        print(video_ids)

        transcribe_youtube_videos(video_ids)
elif youtube_video_ids:
    print("Processing video ids: " + youtube_video_ids)
    initialize_folder(dir_path)
    # Clear channel id session state
    youtube_channel_id = ''
    st.session_state["youtube_channel_id"] = ''
    print(st.session_state["youtube_video_ids"])
    # Delete directory
    if st.session_state["youtube_video_ids"] != youtube_video_ids:
        # Capture new youtube channel id
        st.session_state["youtube_video_ids"] = youtube_video_ids

        videos_ids = youtube_video_ids.split(',')
        transcribe_youtube_videos(videos_ids) 

# # video_ids = os.getenv("VIDEO_IDS")
# video_ids = ['hMEmNM43sbM','42gqvj1NQl4','frHi8NLom5o','hJyaBGpxM3Y','yOpu0MuKQx4','yveAIKv9agQ','tM6LT1_SXWk','8MO4_CT4ejM','cYJDY31ObEI','4BrodSFUxWc','y7J2Pn1ehE4','XMnAZ0ALgDA','Q1AtNDmUkPI','nfwqk-Nho5A','jqkxtLiJYUM','Sjb9LOZzVRw','18z8cMhmTvY','uADe3pvc3YE','BWP7UbDeSyU','iQSOpv8mV-w','Xxa82WYTA_s','8aBxY3FEu_U','zAB39sbuPJM','zau98ocu2Ew','Y64qkxRV7ww','3i2JUBB3fhI','SDIpAPnjg4Q','gmYaKC0u3aQ','oq2noFp6sv8','j5M_C8XeL54','uAnbplXswBs','cpVo-2FNBxw','1YmPWhN7ebE','VgUgJGMkSG8','461hx9hTDKQ','FgLyGBHKDck','o3E2RZbjpCU','5JQKDCtChSg','70uUNfpnWuI','56-e_38yHWg','yEDrDiThyvQ','SSn5b-g-VSw','9A1dUzxehVM','pC9LcRCaImo','JsFxUqZa3hU','C9KdchG2nE4','Qhgo4aRTLQ8','1d_GwxFaL8U','bW28eyxh1uI','9_atwzEeY88','d8FyU00Y4Ro','GcnQbvhzSMY','k-awHr-XyQI','dDI3bTLjxPA','pbeBodBK1EE','2WMnEwo7tqU','koPQYtPDhgg','JpEccEkiOg0','69eXHr7wiww','fONP-oa4X2k','mgQdouqEoBc','mEp7HG-fA1w','AnzMb1voM1c','mh3fCJcslT8','fB4mx9JLjIM','tMzjF_BYrZc','NNwD4TXTmrQ','od-LmasPW9E','szB8szkJuQg','RtxChjImGuc','qbAMm5Uu7K8','xtQ8RqVaHPs','BCxj3KJVYHI','_4D8TmHF6Gc','rkGsVUNq6v8','MyX8xlt9PoQ','MqUjoIlwpdI','bXXjX-8-KoE','R00WZG6SeZw','2qvvPJkLpdE','g_T0ZVJh6lo','ziSVQ2dG2ew','WVvnFcM8ZBM','Qm13Wi0wwQw','vsydURrcSwk','MibzPJ2QrH4','rs4oX_rXxPU','32guJg1RYN4','d-fK35qfCWM','he3HWCBKFqE','AbCM6lt6G9M','KWhT5fUAXuI','TF6PiZVuwgI','Q6wNmjb4PBM','HdgJMfJPXFo','ak2xmwsby9g','VVOrcH3erPs','Y8uny_Ldt8E','KUuUexEXuYo','YG6Ftqy_L5w','vOutpydGRSY','FkaAB0hVy6A','eQMbQzcn9p4','a6GVgIXbALU','kcJHp6oxUTI','XC5rUFAJB-Y','8_9mvc9KpYI','DSEhOvMal4k','8I4dzEuigf8','pDqd70_NUYo','k6d8wUl_yN0','JfpT_xrpxGc','wHrIbCI-s74','_ZBXguo4q1w','CbchIc1EUEQ','NPXDiXTYUfU','IjyxEwqLyIQ','VJUMGWA6PQc','i-ybq1WS_cg','8VoxjefbOlQ','zWmJcLcrvIE','JV9wT_unBkE','s176JzaS4X0','atNFAgAtX1c','DAPjMgGZmm8','06VSFZIpa3M','5tphyFw4Nsc','1CmSljQHtiU','EGGft8Qn3RE','p5jBaZilYvM','9NKyWgfKD6U','420MKN51Mzg','Ktd53fi5X9w','LizCU1TmIE8','-8s2tQNjRow','MWRBbCySLFI','O7o5rIu71JE','Wn_BUKIDY30','Yw73zG8UAb4','56unqSRcyDk','MaChxMkI2Ls','e0XEyQNQPAM','5VGNiJrkGRM','4KSo8Zq-n3s','varB2b3QMQY','Ad8xtOBnRG0','J6LqaoMxpuI','X584ldMl4-U','djlVAM0W8Ck','W5D62G5sXoE','fxCz7Ax4YlU','pmLS2WC9w04','ul6eAKO064g','gn06cM8VsTA','kISYMs3MELQ','kDWAWoY3uhY','Xz37xYdQcls','ppAeZEd8TZ4','SovtqTx-rf8','Y3gAasrERUA','m4Wcx9evhnI','_P1n8Ywlj8c','OyzAPOE58s8','iS4GOZmnjRI','gc3mX99WnaU','XlwrmEL0QnM','Yhcc-YhuC8s','YVe_DDYJPVo','LeTLjicyMfE','oi0VnHRoPek','nINQn5zsg-k','JGSC7Z86dyw','W9_q_K0p2As','uoqNGAhtpj0','vfb-bgLDJqk','iLfCJkAm5lI','2cJEq5Zc6UQ','N8lBuPxRI1w','KlLZtICs3lI','_PsMLj-EGa8','p_Q-Ate-NWs','legm4nuCJDA','JmR3lYW1uyE','BGzovLLxhlA','e0MmpzFpSeE','lvXkoQ0Sbts','0oGedPqWl6o','1empNMh0TaY','qw_N-2Zy-rA','TcD-L7P15r0','JQsjFvHCDX8','Oddyai0qHEM','gkooOn9XoBQ','y84QmfYC4nk','sYCVsWQF-3A','3f1nkkN3pZc','DfllKZPa2Kg','whMAoZBfEMo','MNy5YYwmQC0','pLqv7w7R-4w','iQZiOSbsvZs','yDQM_5LHn0g','2pMfrfAA1Ck','rlTTvMP5dT4','euYO-cdS0js','dUjaDzeN6mo','JySYTTIuPNw','2z-R_tEoiLU','Gn-TIKs_kPw','Tkl1gvETxRo','eqk4Onh7tec','RgMq5-qYRxA','4VLWLynUx0Y','-QZragRMmRw','iQW_AZv5p14','O3wJBoCPsY0','HRRLXBjQPdc','TUsQQrPjeok','ya07uY-iNTU','LxjV26LSKxc','iLlSwZpwZzo','hfLSoOQipVA','vk7stk08nUY','uJX1vxqPq2g','Fk64zP6JrV8','a15Y7ZL3W7M','2zxd4zeLWuM','SXdrJLCex0w','j6XI0mKEN8U','CV3_yRdu5bE','8fr9pTNOKV0','zcdpdDxgLvg','CsnDprkIwWo','JMDYj6jyg3o','QlTf168NEvs','QL_9fr0qFxM','PY1ILrlQ-uM','AdjGDtM-Yy4','Jaeeo1EqZ6M','U7MmPN3MY5w','F3v6EP7cPUA','Qil8H1uquko','5VJ981yB9_M','0iLl594t8jQ','TofEjdJN9z8','KJD7HGPmrZg','xtljIYTIB2k','uk5VaIN2Nnw','aHJyLjXwTyQ','8cQB-Pfoh_U','2YLDIzWmLm4','BEOoD4-RheI','O47KHoB6XO8','06_rTuq6ooA','8wHxGrW6p6o','944rk_qamhw','LcWZrNpEZJE','neWWomYms_I','0fGuwMqLAjs','K8Cia8h_VjY','rSLNz-wHhKY','mykk4wZGd-k','LXwua9gR05s','PhfgaAk8tD8','NY-BDg3A7fM','DHUTkCw07mo','QmmVu2GD0hA','bjXMbYCMSJo','cjj1tUc8keU','j5U1_SVsnlA','8uu3gr_RBcY','qMSY8Q3PSBc','E3EsN0Mo4v0','9CZpcQajd40','hhlGnEPIw-Y','oOR27O2wO94','XQRLVVJA1Ag','R3AYo8TsD5U','UrKVa_QpuPM','rwz2i_7SObQ','4So4Bap3ZzU','xPc5O-tzbPk','V4w6VtOlSAQ','YNKFIEdhbWw','nma4BnV5Jmc','vjnmttnsS3s','bwoivMH89OY','UrkuS1XqB4Q','dOX8HuGcEgI','uLCF3lvoH9g','6k9bKt_tJhc','dD768KkO3lc','wXxw8mrgBbw','3nlT51O4Xf8','y9O0XBALCiI','jjiEAkjncWI','SFCuqOejy58','mmgXFbcH6N4','tHLzdqsI7Ko','HVfDgANVhe0','LlwGJml9wJw','53gymcla0Ro','rShJkDU3kjQ','4BTKOgsqyqc','LUgiT-wXDLE','6zUfdlYu-50','GhbBFWBuF7I','86116qrMoos','FQm8OC-WnBY','I5eVtoyUB6c','lCEgbem2vFc','wXnpULiKyno','0K631zduL_o','Jj86Eal2HIM','xD1Jj5WCP1k','lNe7r2w6ROk','dTlx4nHXhXU','9v-ll0757Ks','xneMHX5nV9k','LlOZq_M5dlg','enh__F5hAbg','TwB5RNIiO5o','NvL5ftRl0ug','GGS4ejvplAQ','0ikUNvGKhHc','soLFNf-myZs','BPGoaXYfLe8','ulxyA2zx8G4','L_UI5CkAJxs','iKj4oe760e0','EPsaQkpSU3I','_IEsEF2RxZk','3g-1L6pluHM','jZzXBdrYSYU','oLSZiC12hnk','WtK_CRBdZTA','uYGiQc_wF40','TvFDHekuxIc','CnQhxFeaef0','5VRizVpFaPw','6mR4_B3GkYI','E9a_aLuJQ04','qkLFolAIOKA','GgL7fR53A7w','VelbzA51ILI','B7MoLRhOzn8','xK64djBQigs','pO4NJkZUqyE','9FD_IKKgcxw','6Lcu7RyHndk','KPjooxto8f8','PMRzd875N9I','rkvKSDzd4Gc','YaRT-tAYV7I','jGU176HX7UM','IOlUge28QTE','k4ekywRhcQ0','nR050GhfYGA','A4ZGQ7lO_HE','jGWU37mrQiA','mnBSRyn75ng','KQdoQSkCNoY','dh9_THubhWM','KS3q5cffdzs','w6oVO9eY0uU','mdyi7HktA0I','Tqcyc8QJGeQ','6kimbbxqrno','jOS5zqKt7fE','BjSljempuIQ','W5RMXRZnsf0','sM0qUIbSTIk','Aly3t30np_Q','fMMfoj1bWn0','TdYRWPZ2VZs','5ahHgNXDrI4','UMvsfgUkeik','juczwkJWzbE','Z1fmyyPtj6I','Gkhqh5Fb25g','Eo17LD0NPQk','PYS69W2MOGo','XAIBccSlaFQ','n2wopdvIHyc','Mlb8XCdeVlM','bxkiTADqac4','EqyMvIzGhgU','Id2HR2y6-XE']
# print(video_ids)

# Initiate Transcription
# process_pool = multiprocessing.Pool(processes=4)
# result = process_pool.map(get_youtube_video_transcript, video_ids)
# for video_id in video_ids:
#     get_youtube_video_transcript(video_id)

qa_chain = configure_qa_chain()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text

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
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.container.write(f"**Document {idx} from {source}**")
            self.container.markdown(doc.page_content)

# New Code
def generate_and_play(audio_text):
    print("Generating")
    set_api_key(ELEVENLABS_API_KEY)
    # Generate audio using ElevenLabs
    audio = generate(text=audio_text, voice=getVoice("Alli"), 
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

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())

        try:
            response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
            st.session_state.messages.append({"role": "assistant", "content": response})
            # print(response + str(ENABLE_ELEVENLABS) + str(FINE_TUNE_VOICES))
        
            #Enable or Disable Eleven Labs
            if ENABLE_ELEVENLABS:
                print('Hello.. Eleven Labs')
                condensed_answer = response[0:2]
                print(condensed_answer)
                # New Code
                # Use ElevenLabs API to generate speech and play it
                if ELEVENLABS_API_KEY:
                    generate_and_play(audio_text=condensed_answer)
                else:
                    st.error("ElevenLabs API key not found. Please set the 'ELEVENLABS_API_KEY' environment variable.")

            st.write(response)
        except Exception as ex:
            print(ex)
            st.write("An unexpected error has occured.")