import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'message_handler'))

from dbert_train import DBertTrainer
from stt import STT
from simpleNLP import SimpleNLP

def training():
    trainer = DBertTrainer(dataset_path="train/datasets.csv")
    trainer.train()

def start_stt_process():
    stt = STT()
    text = stt.transcribe_audio(file_path="audio_files/recorded.wav")
    print(text)
    
def start_nlp_process():
    nlp = SimpleNLP()
    result = nlp.handle_audio("audio_files/recorded.wav")
    print(result)

if __name__ == "__main__":
    # training()
    # start_stt_process()
    start_nlp_process()