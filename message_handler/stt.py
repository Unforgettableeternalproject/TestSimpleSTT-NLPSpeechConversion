import speech_recognition as sr

# Function to transcribe audio file

class STT:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
    
    def realtime_stt_process(self, text_queue):
        print("Starting real-time STT. Speak into the microphone...")
    
        # Adjust for ambient noise levels
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Calibrated for ambient noise. You can start speaking now.")

        while True:
            try:
                with self.mic as source:
                    print("Listening...")
                    audio = self.recognizer.listen(source)
            
                print("Transcribing...")
                text = self.recognizer.recognize_google(audio)
                print(f"Transcribed Text: {text}")

                # Place the text in the queue for NLP processing
                text_queue.put(text)
        
            except sr.UnknownValueError:
                print("Could not understand audio. Please try again.")
            except sr.RequestError as e:
                print(f"API request error: {e}")
            except KeyboardInterrupt:
                print("Stopping real-time STT.")
                break

    def debug_stt(self, file_path):
        with sr.AudioFile(file_path) as source:
            audio = self.recognizer.record(source)
        try:
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError as e:
            return f"API request error: {e}"
