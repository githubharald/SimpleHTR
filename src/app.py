# We import Flask
from flask import Flask
from src.main import result
from googletrans import Translator

# We create a Flask app
app = Flask(__name__)


# We establish a Flask route so that we can serve HTTP traffic on that route
@app.route('/')
def weather():
    reco, proba = result('../data/test.png')
    tr = Translator()
    s = tr.translate(reco, dest='fr', src='en')
    print("\nTraduction: "+str(s.text))
    # We hardcode some information to be returned
    # return "{'Temperature': '50'}"
    return "Recognized: \""+str(reco)+"" \
        "\" \nProbability: "+str(proba)+\
           "\nTraduction: "+str(s.text)


# Get setup so that if we call the app directly (and it isn't being imported elsewhere)
# it will then run the app with the debug mode as True
# More info - https://docs.python.org/3/library/__main__.html
if __name__ == '__main__':
    app.run(debug=True)