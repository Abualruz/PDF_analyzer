from msilib.schema import File
from flask import Flask,render_template,request,redirect,url_for,Response,jsonify
from flask_wtf import FlaskForm
from flask import render_template
from wtforms import FileField,SubmitField
from werkzeug.utils import secure_filename
import os
from pathlib import Path
import cv2
import logging

#UPLOAD_FOLDER =''
ALLOWED_EXTENSIONS= {'pdf'}

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
try:

    path = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER=os.path.join(path.replace("/file_folder",""))
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SECRET_KEY']='supersecretkey'
    class UploadFileForm(FlaskForm):
        file=FileField("File")
        submit=SubmitField("UPLOAD_FOLDER")
except Exception as e:
    app.logger.info('An error occurred while creating temp folder')
    app.logger.error('Exception Occurred :{}'.format(e))


@app.route('/', methods=['GET','POST'])
@app.route('/home', methods=['GET','POST'])
def home():
    form=UploadFileForm()
    if form.validate_on_submit():
        file=form.file.data # grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename)))
        return "file has been uploaded"

    return render_template('index.html',form=form)


if __name__ == '__main__':
    app.run(debug=True)        