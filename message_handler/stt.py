import speech_recognition as sr

# Function to transcribe audio file

class STT:
    def __init__(self):
        pass

    def transcribe_audio(self, file_path):
        recognizer = sr.Recognizer()
        with sr.AudioFile(file_path) as source:
            audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"API request error: {e}"
