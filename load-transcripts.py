import os
import pinecone 
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import scrapetube
from dotenv import load_dotenv
import tomllib

load_dotenv()

def load_local_environment_variables():
    with open(".streamlit/secrets.toml", "rb") as f:
        return tomllib.load(f)

secrets = load_local_environment_variables()

# print(secrets)

PINECONE_API_KEY = secrets['PINECONE_API_KEY']
PINECONE_ENV = secrets['PINECONE_ENVIRONMENT']
PINECONE_INDEX = secrets['PINECONE_INDEX']
OPENAI_API_KEY = secrets['OPENAI_API_KEY']
LIMIT_NUMBER_OF_VIDEOS = eval(str(secrets['LIMIT_NUMBER_OF_VIDEOS']))
MAX_VIDEO_COUNT = secrets['MAX_VIDEO_COUNT']
# print(PINECONE_API_KEY)
# print(PINECONE_ENV)
# print(PINECONE_INDEX)
# print(OPENAI_API_KEY)


# PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
# PINECONE_ENV = os.getenv('PINECONE_ENV')
# PINECONE_INDEX = os.getenv('PINECONE_INDEX')
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# print(OPENAI_API_KEY)

# House Keeping
dir_path = "transcript_files"

# Create a list to store the names of the generated .txt files
txt_files = []

# Clear Dir
def clear_directory(dir_path):
  for file in os.scandir(dir_path):
    print(file.path)
    os.remove(file.path)

def doc_preprocessing():
    loader = DirectoryLoader(
        dir_path,
        glob='**/*.txt',     # Only the Txt files
        show_progress=False
    )
    docs = loader.load()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)

    docs_split = text_splitter.split_documents(docs)
    return docs_split

# @st.cache_resource
def create_embeddings():
    # we use the openAI embedding model
    embeddings = OpenAIEmbeddings(
       openai_api_key=OPENAI_API_KEY
    )

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

# Transcribe video ids
def transcribe_youtube_videos(videos_ids):
    i=0
    for video_id in videos_ids:
        i+=1
        print('Transcribing Video ID: ' + video_id)
        get_youtube_video_transcript(video_id)

        if LIMIT_NUMBER_OF_VIDEOS:   
            if i>=MAX_VIDEO_COUNT:
                break

# Initialize
def initialize_folder(dir_path):
    if os.path.isdir(dir_path):
        print("Folder Exists.. Deleting the folder")
        clear_directory(dir_path)
        os.removedirs(dir_path)

    # Create the directory for new content
    os.makedirs(dir_path)

def main():
    initialize_folder(dir_path)
    videos = scrapetube.get_channel(channel_username="NathanLatkawatch")
    # print(videos)
    video_ids = [video['videoId'] for video in videos if video is not None]
    print(video_ids)
    transcribe_youtube_videos(video_ids)
    create_embeddings()
    print("Embeddings completed")

if __name__ == "__main__":
    main()