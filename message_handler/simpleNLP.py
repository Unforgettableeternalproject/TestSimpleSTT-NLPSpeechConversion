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
        return "command" if predictions[0] == 0 else "chat"

    def handle_audio(self, file_path):
        # Step 1: Transcribe audio
        text = self.stt.transcribe_audio(file_path)
        
        # Step 2: Classify message as command or chat
        message_type = self.classify_message(text)
        
        # Step 3: Take action based on message type
        if message_type == "command":
            return f"Executing command based on: '{text}'"
        else:
            return f"Chat response: '{text}'"
