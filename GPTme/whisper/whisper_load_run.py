import sys
import os
import shutil
import subprocess
from GPTme.llms.hf_load_model import download_model
from GPTme.config import DEVICE, WHISPER_PATH, WHISPER_INPUT, WHISPER_OUTPUT, ROOT, WHIPER_MODELS

def git(*args):
    return subprocess.check_call(['git'] + list(args))

whisper_map = {
    "tiny":"ggml-tiny.en.bin",
    "small":"ggml-small.bin",
    "medium":"ggml-medium.bin",
    "medium_q5":"ggml-medium-q5_0.bin",
    "base":"ggml-base.bin",
    "base_q5":"ggml-base.en-q5_1.bin",
    "base_en":"ggml-base.en.bin",
    "large_v3_q5":"ggml-large-v3-q5_0.bin",
    "large_v3":"ggml-large-v3.bin",
}

def download_whisper(model_type="tiny",model_base="") -> str:
    # This function downloads the Whisper.CPP model from the HF repo
    model_path = ""
    try:
        if model_type in whisper_map:
            if model_base == "":
                model_base = whisper_map[model_type]
            else:
                if model_base != whisper_map[model_type]:
                    print("Error - the provided parmeters for Whisper model don't match please use either model_type or model_base")
        else:
            print("Error - the provided parmeters for Whisper model type isn't mapped use model_base")
        # Files can be found here - https://huggingface.co/ggerganov/whisper.cpp/tree/main
        model_path = download_model(model_id="ggerganov/whisper.cpp", model_base=model_base, model_path=WHIPER_MODELS)
    except:
        print(f"Error - Couldn't load the specified model - {model_base}")
    finally:
        return model_path
    
def init_whisper():
    # Erase all previous audio & output files
    if os.path.isdir(WHISPER_INPUT): 
        shutil.rmtree(WHISPER_INPUT)
        
    if os.path.isdir(WHISPER_OUTPUT): 
        shutil.rmtree(WHISPER_OUTPUT)

    os.mkdir(WHISPER_INPUT)    
    os.mkdir(WHISPER_OUTPUT)

def build_whisper(device=DEVICE):
    # This function downloads the WhisperCPP repo and builds it according to the supported device ('cpu' vs. 'cuda')
    try:
        if not os.path.isdir(WHISPER_PATH):
            os.chdir(os.path.join(ROOT,"../"))
            git("clone", "https://github.com/ggerganov/whisper.cpp.git")
    except:
        print("Error couldn't clone Whiper Repo - check internet connection")
        return False
    os.chdir(WHISPER_PATH)
    subprocess.run("make clean", shell=True, stdout=subprocess.PIPE)
    try:
        print("Building Whisper")
        if device == 'cuda':
            subprocess.Popen("WHISPER_CUBLAS=1 make -j", shell=True, stdout=subprocess.PIPE)
        else:
            subprocess.Popen("make -j", shell=True, stdout=subprocess.PIPE)
    except:
        print("Error while building Whisper")    
        sys.exit(1)  

def decode_audio(in_filename, **input_kwargs):
    new_file_path = ""
    # The function converts MP3 files to WAV in 16K PCM single channel sample rate
    try:
        exit_code = subprocess.check_output(["ffmpeg", "-version"])
    except :
        print("Error - might need to install ffmpeg - run - 'sudo apt install ffmpeg'")
        return sys.exit(1)
    try:
        base_name, _ = os.path.splitext(in_filename)
        new_file_path = base_name + "_converted." + "wav"
        
        print(f"decoding the audio file {in_filename} to {new_file_path}")
        
        sampleRate = '16000'
        numberChannels = '1'
        format = 'pcm_s16le'
        args = ['-i', in_filename, '-ar', sampleRate, '-ac', numberChannels, '-c:a', format, new_file_path]
        # Based on the command from LlamaCPP docs - ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
        # ffmpeg -i inputFile -ar 16000 -ac 2 -c:a pcm_s16le outputFile
        subprocess.call(['ffmpeg'] + list(args))
    except:
        sys.exit(1)
    
    return new_file_path

def run_whisper(model_path,input_file):
    try:
        base_dir_name = os.path.split(input_file)
        transcript_file = base_dir_name[1].split('.')[0]
        cmd = ["./main", "-m", model_path, "-f", os.path.join(WHISPER_INPUT,base_dir_name[1]), "-otxt", "-tr", "-of", os.path.join(WHISPER_OUTPUT,transcript_file)]

        ### DEBUG prints ###
        # print(f"input file - {input_file}")
        # print(f"input file to list - {base_dir_name}")
        # print(f"Transcript file - {transcript_file}")
        # print(f"Whisper path {WHISPER_PATH}")
        # print(f"Whisper input {WHISPER_INPUT}")
        # print(f"Whisper output {WHISPER_OUTPUT}")
        # print(f"Sending to whisper - {os.path.join(WHISPER_INPUT,base_dir_name[1])}")
        # print(f"Getting from whisper - {os.path.join(WHISPER_OUTPUT,transcript_file)}")
        # print(f"Whisper command - {cmd}")

        os.chdir(WHISPER_PATH)
        subprocess.run(cmd)#,shell=True, check=True,capture_output = True, text = True)

    except:
        sys.exit(1)


# run_whisper(decode_audio('./audio/question_2.wav'))


