from flask import Flask, render_template, request, send_file
import os
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = '../data/page'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    # Access the uploaded file using the 'request' object
    uploaded_file = request.files['file']

    if uploaded_file:
        # Save the uploaded file to a specific location
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

        # Run your transcription process here and save the result to "transcribed.txt"
        streamline_path = '../streamline.py'
        subprocess.run(['python', streamline_path])

        # Read the content of "transcribed.txt"
        transcribed_file_path = './transcribed.txt'
        with open(transcribed_file_path, 'r') as transcribed_file:
            transcribed_text = transcribed_file.read()
        os.remove(file_path)

        # Return the content as a response
        return transcribed_text

    else:
        # Return an error response if no file was uploaded
        return 'No file uploaded', 400
    
@app.route('/output.mp3')
def returnAudioFile():
    audio_file_path = 'C:/Users/Shakib/Desktop/ai camp/projects/Detect-Transcribe/website/output.mp3'
    return send_file(
         audio_file_path, 
         mimetype="audio/mpeg", 
         as_attachment=True, 
         download_name="output.mp3")
    
if __name__ == '__main__':
    app.run()