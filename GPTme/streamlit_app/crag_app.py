import streamlit as st

from GPTme.pipes.local_crag import run_crag_app
from GPTme.ingest.database import get_docs_list, create_db, clear_db
from GPTme.whisper.whisper_load_run import download_whisper, build_whisper, decode_audio, run_whisper, init_whisper
from GPTme.config import SOURCES_PATH, DB_PATH
import os
from audiorecorder import audiorecorder
from GPTme.config import WHISPER_INPUT, WHISPER_OUTPUT

if 'init' not in st.session_state:
    st.session_state.init = False

if 'question' not in st.session_state:
    st.session_state.question = ""

if 'qNum' not in st.session_state:
    st.session_state.qNum = 0

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

if not st.session_state.init:
    st.session_state.init = True
    init_whisper()
    download_whisper(model_type='tiny')
    build_whisper()

    current_file_list = get_docs_list()
    state = 'Pending DB'
    database_ready = False

def reset_db():
    global state
    global database_ready
    database_ready = False
    clear_db()
    state = 'Rebuilding DB - Please hold'
    create_db()
    state = 'Ready for you questions'
    database_ready = True

def generate_response(prompt_input):                    
    return run_crag_app(question=prompt_input)

with st.sidebar:
    st.title("Local Corrective RAG with your Data")
    "[View the source code](https://github.com/shalvamist/GPTme/blob/main/GPTme/streamlit_app/crag_app.py)"
    # Loading the file
    uploaded_files = st.file_uploader("Upload an article", 
                                    accept_multiple_files=True,
                                    type=('txt', 'pdf', 'doc', 'docx', 'py', 'xls', 'csv'),
                                    )
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            save_path = os.path.join(SOURCES_PATH, uploaded_file.name)
            with open(save_path, mode='wb') as w:
                w.write(uploaded_file.getvalue())
            if os.path.isfile(save_path):
                print(f'File {uploaded_file.name} is successfully saved!')
        database_ready = False
        state = 'Rebuilding DB - Please hold'
        create_db()
        state = 'Ready for you questions'
        database_ready = True
    st.button("Reset Database", on_click=reset_db)
    st.write(f'## Click here to recored your question')
    audio = audiorecorder("Record a question", "Stop")

    if len(audio) > 0:
        # To play audio in frontend:
        st.audio(audio.export().read())  
        # To save audio to a file, use pydub export method:
        # st.write(f"### question number - {st.session_state.qNum}")
        question_audio_file = os.path.join(WHISPER_INPUT,"question_"+str(st.session_state.qNum)+".wav")
        audio.export(question_audio_file, format="wav")
        # To get audio properties, use pydub AudioSegment properties:
        st.write(f"Frame rate: {audio.frame_rate}, Frame width: {audio.frame_width}, Duration: {audio.duration_seconds} seconds")
        st.session_state.qNum += 1
        run_whisper(decode_audio(question_audio_file))
        base_dir_name = os.path.split(question_audio_file)
        transcript_file = base_dir_name[1].split('.')[0] + '_converted.txt'
        transcript_file = os.path.join(WHISPER_OUTPUT,transcript_file)
        with open(transcript_file) as f:
            st.session_state.question = f.read().replace('\n', '')
        st.session_state.messages.append({"role": "user", "content": st.session_state.question})

st.title("📝 Q&A with Local CRAG")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_response(prompt) 
            st.write(response) 
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
