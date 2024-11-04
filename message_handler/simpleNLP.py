from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from stt import STT

class SimpleNLP:
    def __init__(self, model_dir="./command_chat_classifier"):
        self.stt = STT()
        self.model_dir = model_dir
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)

    def classify_message(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
        
        # print(predictions.item())
        label_mapping = {0: "command", 1: "chat", 2: "non-sense"}
        return label_mapping[predictions.item()]

    def debug_nlp(self, file_path=None):
        if(file_path is not None): text = self.stt.debug_stt(file_path)
        else: text = input("Enter text: ")
        
        message_type = self.classify_message(text)
        
        match(message_type):
            case "command":
                return f"'{text}' is command."
            case "chat":
                return f"'{text}' is chit-chat."
            case "non-sense":
                return f"'{text}' is non-sense."
            case _:
                return "Unknown message type."
        
    def process_transcriptions(self, text_queue):
        while True:
            text = text_queue.get()
            
            if text is None:
                break
            
            message_type = self.classify_message(text)
            
            if message_type == "command":
                print(f"Command: {text}")
            else:
                print(f"Chit-chat: {text}")
