from flask import Flask
from flask import request, Response
#import jsonpickle

app = Flask(__name__)

@app.route("/")
def index():
    return (
        """<form action="upload" method="post" id="upload-form">
        <input type="file" name="imagefile" id="imagefile"/>
        <input type="submit" />
        </form>"""
    )

@app.route('/upload', methods=['POST'])
def upload():
    try:
        return "response try I can say hello"
    except Exception as e:
        app.logger.exception(e)
        return "Can't say hello."

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)