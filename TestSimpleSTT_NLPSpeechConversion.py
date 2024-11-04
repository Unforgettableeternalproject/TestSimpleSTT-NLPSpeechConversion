import sys
import os
import threading
import queue

sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'message_handler'))

from dbert_train import DBertTrainer
from stt import STT
from simpleNLP import SimpleNLP

def training():
    trainer = DBertTrainer(dataset_path="train/datasets.csv")
    trainer.train()

def start_stt_debug_process():
    stt = STT()
    text = stt.debug_stt(file_path="audio_files/recorded.wav")
    print(text)
    
def start_nlp_debug_process():
    nlp = SimpleNLP()
    #result = nlp.debug_nlp("audio_files/recorded.wav")
    result2 = nlp.debug_nlp()
    print(result2)
    
def realtime_stt():
    stt = STT()
    nlp = SimpleNLP()
    text_Queue = queue.Queue()
    
    stt_thread = threading.Thread(target=stt.realtime_stt_process, args=(text_Queue,))
    nlp_thread = threading.Thread(target=nlp.process_transcriptions, args=(text_Queue,))
    
    stt_thread.start()
    nlp_thread.start()

if __name__ == "__main__":
    # training()
    # start_stt_debug_process()
    try:
        while True:
            start_nlp_debug_process()
    except KeyboardInterrupt:
        print("keyboardinterrupt")
    #realtime_stt()