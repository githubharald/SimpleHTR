import os
import main as transcribe
import detect 
import playsound

current_file = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]= os.path.join(current_directory, "key.json")
from google.cloud import texttospeech

def main():
    detect.main()
    transcribe.main()

    #reads file that contains the transcribed sentence
    transcribed_text_file = os.path.join('./', 'transcribed.txt')
    with open(transcribed_text_file) as f:
        transcribed_text = f.read()
        

    client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=transcribed_text)

    voice = texttospeech.VoiceSelectionParams(
    language_code='en-US',
    name='en-US-Wavenet-C',
    ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)

    audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3)

    response = client.synthesize_speech(
    input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open('output.mp3', 'wb') as out:
    # Write the response to the output file.
        out.write(response.audio_content)
        print('Audio content written to file "output.mp3"')


if __name__ == '__main__':
    main()
#playsound('./output.mp3')