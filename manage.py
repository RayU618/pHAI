from torchvision import datasets, models, transforms
from flask import Flask, render_template, request, flash
from wtforms import Form, FloatField, SubmitField, validators, ValidationError
import numpy as np
#from sklearn.externals 
import joblib
import cv2
import datetime
import torch
import torch.nn as nn
import tensorflow as tf
 
# 学習済みモデルを読み込み利用します
def predict(parameters):
    # ニューラルネットワークのモデルを読み込み
    device = torch.device("cpu")
    model_ft = models.resnet18(pretrained=False).to(torch.device('cpu'))
    model_ft = model_ft.to('cpu')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 8)
    model_ft.load_state_dict(torch.load('./ResNet.pkl'))
    params = torch.Tensor(parameters)
    params = torch.reshape(params, (1, 3, 224, 224))
    #params = parameters.reshape(-1,1)
    #pred = model.predict(params)
    pred = model_ft(params)
    return torch.argmax(pred).numpy()
 
# ラベルからIrisの名前を取得します
def getName(label):
    print(label)
    if label == 0:
        return "pH 6.00 ~ 6.25"
    elif label == 1: 
        return "pH 6.26 ~ 6.50"
    elif label == 2: 
        return "pH 6.51 ~ 6.75"
    elif label == 3: 
        return "pH 6.76 ~ 7.00"
    elif label == 4: 
        return "pH 7.01 ~ 7.25"
    elif label == 5: 
        return "pH 7.26 ~ 7.50"
    elif label == 6: 
        return "pH 7.51 ~ 7.75"
    elif label == 7: 
        return "pH 6.76 ~ 8.00"
    else : 
        return "Error"
 
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'zJe09C5c3tMf5FnNL09C5d6SAzZoY'
 
# 公式サイト
# http://wtforms.simplecodes.com/docs/0.6/fields.html
# Flaskとwtformsを使い、index.html側で表示させるフォームを構築します。
class IrisForm(Form):
    """
    SepalLength = FloatField("Sepal Length(cm)（蕚の長さ）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
 
    SepalWidth  = FloatField("Sepal Width(cm)（蕚の幅）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
 
    PetalLength = FloatField("Petal length(cm)（花弁の長さ）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
 
    PetalWidth  = FloatField("petal Width(cm)（花弁の幅）",
                     [validators.InputRequired("この項目は入力必須です"),
                     validators.NumberRange(min=0, max=10)])
    """
 
    # html側で表示するsubmitボタンの表示
    submit = SubmitField("判定")
 
@app.route('/', methods = ['GET', 'POST'])
def predicts():

    img_dir = "static/imgs/"
    if request.method == 'GET': img_path=None
    elif request.method == 'POST':
        #### POSTにより受け取った画像を読み込む
        stream = request.files['img'].stream
        img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        img_tensor = tf.multiply(img_array, 1)
        #### 現在時刻を名前として「imgs/」に保存する
        dt_now = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        img_path = img_dir + dt_now + ".jpg"
        cv2.imwrite(img_path, img)
        img = cv2.resize(img , (224, 224))
        pred = predict(img)
        #pred = predict(img_tensor)
        irisName = getName(pred)
        #irisName = pred
        return render_template('result.html', irisName=irisName)
    #### 保存した画像ファイルのpathをHTMLに渡す
    return render_template('index.html', img_path=img_path) 

    """
    form = IrisForm(request.form)
    if request.method == 'POST':
        if form.validate() == False:
            flash("全て入力する必要があります。")
            return render_template('index.html', form=form)
        else:            
            SepalLength = float(request.form["SepalLength"])            
            SepalWidth  = float(request.form["SepalWidth"])            
            PetalLength = float(request.form["PetalLength"])            
            PetalWidth  = float(request.form["PetalWidth"])

            #x = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth])
            pred = predict(x)
            irisName = getName(pred)
 
            return render_template('result.html', irisName=irisName)
    elif request.method == 'GET':
 
        return render_template('index.html', form=form)
    """
 
if __name__ == "__main__":
    app.run()
 
