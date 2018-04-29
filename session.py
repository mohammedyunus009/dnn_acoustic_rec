from flask import Flask, session, render_template, request, redirect, g, url_for
import os
import uuid

app = Flask(__name__)
app.secret_key = os.urandom(24)


__author__ = 'ibininja'
#app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
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

    #return ("Files has been saved successfully to: " + destination_file_path + " you can further use this message to communicate any message to front'end"+msg)


@app.route('/protected',methods=['GET', 'POST'])
def protected():
    print("Inside...")

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if g.user:
        print("user is yes")
    else:
        print("user is no")

    print("method: " + request.method)

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    if g.user:
        print "done1"
        upload()
        # predict()
        print "done2"
        return render_template('pro.html')

    return redirect(url_for('index'))

def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

    return #render_template("complete.html")

def predict():
    import random
    a=random.randint(0,14)
    with open('templates/pro.html','w') as f:
        f.write("u are in "+labels[a]+" environment")
    print "done3"
    return

@app.before_request
def before_request():
    g.user = None
    if 'user' in session:
        g.user = session['user']

@app.route('/getsession')
def getsession():
    if 'user' in session:
        return session['user']

    return 'Not logged in!'

@app.route('/dropsession')
def dropsession():
    session.pop('user', None)
    return 'Dropped!'

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
