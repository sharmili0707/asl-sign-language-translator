# speak_text.py

import pyttsx3

def speak_text(text):
    """
    Function to convert given text into speech using pyttsx3.
    You can integrate this with your sign language translator project.
    """
    engine = pyttsx3.init()

    # Set properties
    engine.setProperty('rate', 150)   # Speed (words per minute)
    engine.setProperty('volume', 1.0) # Volume (0.0 to 1.0)

    # Optional: Choose voice (0 = male, 1 = female; may vary by OS)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)

    # Speak the text
    engine.say(text)
    engine.runAndWait()


# ✅ Example usage (you can remove this part if importing elsewhere)
if __name__ == "__main__":
    sample_text = "Hello! This is your sign language translator speaking."
    speak_text(sample_text)

