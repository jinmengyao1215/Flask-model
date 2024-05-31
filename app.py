import io
import threading

import numpy as np
import base64
from flask import Flask, request, render_template, send_file
import pickle
import shap
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


app = Flask(__name__)  # 初始化APP

model = pickle.load(open("xgbmodel.pkl", "rb"))  # 加载模型


@app.route("/")  # 装饰器
def home():
    plot_ural = "./static/show.png"
    return render_template("index.html", plot_ural = plot_ural )  # 先引入index.html，同时根据后面传入的参数，对html进行修改渲染。


@app.route("/predict", methods=["POST"])
def predict():
    int_features = [float(x) for x in request.form.values()]  # 存储用户输入的参数

    final_features = [np.array(int_features)]  # 将用户输入的值转化为一个数组
    X = final_features[0].tolist()
    prediction = model.predict(final_features)  # 输入模型进行预测
    prediction_proba = model.predict_proba(final_features)

    output = prediction[0]  # 将预测值传入output

    out = prediction_proba[0]
    out = out[1]

    countries = ["Lower risk of readmission for patients", "Higher risk of readmission for patients!"]

    plot_ural = waterfall(X,model)
    # img_stream = waterfall(listX,model)





    return render_template(
        "index.html",
        risk="{}".format(out),plot_ural = plot_ural
    )




def waterfall(X, model_shap):
    X = [X]
    X = pd.DataFrame(X, columns=['AGE', 'Temperature', 'SBP', 'DBP', 'Neutrophils', 'iP', 'WBC', 'RBC', 'SUA', 'Mg', 'TRAb', 'Hypertension'])
    explainer = shap.Explainer(model_shap)
    shap_values = explainer(X)


    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    plt.tight_layout()

    img_stream = io.BytesIO()
    plt.savefig(img_stream, format='png')
    img_stream.seek(0)

    plot_url = base64.b64encode(img_stream.getvalue()).decode('utf-8')
    plot_url = "data:image/png;base64, "+plot_url


    return plot_url

if __name__ == "__main__":
    import signal

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    app.run(app.run(debug=True, host='0.0.0.0', port=5000))
    # threading.Thread(target=app.run, kwargs={'debug': True}).start()
   # threading.Thread(target=app.run, kwargs={'debug': True}).start()