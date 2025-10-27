from flask import Flask, request, render_template, url_for, redirect
import pickle, numpy


app = Flask(__name__)
with open(file="model.pkl", mode="rb") as f:
    model = pickle.load(f)


@app.route("/")
def welcome():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # get the data from the form
        sl=float(request.form["sepal-length"])
        sw=float(request.form["sepal-width"])
        pl=float(request.form["petal-length"])
        pw=float(request.form["petal-width"])


        inp=numpy.array([sl,sw,pl,pw])
        inp = inp.reshape([1,-1])
        pred=model.predict(inp)
        return str(pred[0])
    return redirect(url_for("welcome"))


if __name__=="__main__":#debug is kept true only on the development server, not on the production server
    app.run(host="0.0.0.0", port=5000,debug=True)
