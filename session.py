from flask import Flask, session, render_template, request, redirect, g, url_for
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)


__author__ = 'ibininja'
#app = Flask(__name__)
#APP_ROOT = os.path.dirname(os.path.abspath(__file__))
labels = ['bus', 'cafe/restaurant', 'car', 'city_center', 'forest_path',
           'grocery_store', 'home', 'beach', 'library', 'metro_station',
           'office', 'residential_area', 'train', 'tram', 'park']


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session.pop('user', None)

        if request.form['password'] == '':
            session['user'] = request.form['username']
            return render_template('upload.html')

    return render_template('index.html')


#Below method is rest request

WAV_DROP_DIR = "audio-files"

@app.route('/upload_file', methods=['POST'])
def upload_file():

    if not os.path.isdir(WAV_DROP_DIR):
        os.mkdir(WAV_DROP_DIR)

    f = request.files['uploadedFile']
    destination_file_path = os.path.join(WAV_DROP_DIR, str(uuid.uuid4().hex) + ".wav")

    print("Destination file path: " + destination_file_path)
    f.save(destination_file_path)
    import production
    msg = production.make_pred(destination_file_path)
    return "This is a " + msg + " sound"


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)