from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import utils

app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/classify_sentiments', methods=['GET', 'POST'])
def tweeter_senti():
    if request.method == 'POST':
        senti = request.form['sentence']
        pred = utils.making_prediction(senti)
        pred1 = np.argmax(pred[0])
        pred2 = np.argmax(pred[1])
        output1 = ""
        output2 = ""
        if pred1 == 0:
            output1 = "negatif"
        elif pred1 == 1:
            output1 = "pozitif"

        if pred2 == 0:
            output2 = "dunya"
        elif pred2 == 1:
            output2 = "ekonomi"
        elif pred2 == 2:
            output2 = "kultur"
        elif pred2 == 3:
            output2 = "saglik"
        elif pred2 == 4:
            output2 = "siyaset"
        elif pred2 == 5:
            output2 = "spor"
        elif pred2 == 6:
            output2 = "teknoloji"

        return render_template("index.html", message=f' {output1} '.upper(), sinif=f" {output2} ".upper() , sentence=senti)


if __name__ == '__main__':
    app.run()
