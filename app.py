import os
import glob
from scipy.special import softmax
from datetime import datetime
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, flash
from werkzeug.utils import secure_filename
from multiprocessing import Process
from gevent.pywsgi import WSGIServer
import matplotlib.pyplot as plt
from controllers import split, image, audio, text, plot


UPLOAD_FOLDER = './data/video_file/'
ALLOWED_EXTENSIONS = {'mp4', 'm4v', 'avi'}
date = datetime.now()


# Declare a flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print('Model loaded. Check http://127.0.0.1:5000/')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('base.html')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('predict'))


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    # we cant delete the plots at the end. so considering that we decided to delete those plots at the start of next process

    plot_path = glob.glob('./static/images/*')
    for p in plot_path:
        os.remove(p)

    p1 = Process(target=split.audio_split())
    p1.start()

    p2 = Process(target=split.image_split())
    p2.start()

    audio_val = audio.audio()
    image_val, frame_time, full_time = image.image()
    text_val = text.text()

    image_soft = softmax(image_val, axis=0)
    audio_soft = softmax(audio_val, axis=0)
    text_soft = softmax(text_val, axis=0)

    image_final = image_soft * image_val
    audio_final = audio_soft * audio_val
    text_final = text_soft * text_val

    final_score = (image_final + audio_final + text_final) / 3
    final_score = round(final_score, 1)

    # plot the image
    plot_url = './static/images/plot{}.png'.format(date)
    score_list = plot.image_plot()
    plt.xlabel("Time duration in seconds")
    plt.ylabel('violation score for every second ')
    plt.plot(score_list)

    plt.savefig(plot_url)

    # remove the content of all the directories

    image_path = './data/image_data/*'
    audio_path = './data/audio_data/audio.wav'
    split_path = './data/audio_data/split_audio/*'
    spec_path = './data/audio_data/spectogram/*'

   # path_list = [image_path, audio_path, split_path, spec_path]
    #for path in path_list:
     #   for f in glob.glob(path):
      #      os.remove(f)


    return render_template('index.html', output1=final_score, output2=frame_time, output3=full_time, url=plot_url)


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)
    Debug =True
    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()

