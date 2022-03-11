from flask import Flask, render_template, request, flash
from joblib import load
from dicts import outputs, country_codes
import  Data_pre_processing


pipeline = load("tweet_dialect.joblib")

app = Flask(__name__)
app.secret_key = "manbearpig_MUDMAN888"

@app.route("/")
def index():
	flash("Tweet")
	return render_template("index.html")

@app.route("/greet", methods=['POST', 'GET'])
def greeter():
	pocessed_text = Data_pre_processing.preprocess(str(request.form['name_input']))
	temp = pipeline.predict([pocessed_text])
	flash(country_codes[outputs[temp[0]]])  
	return render_template("index.html")
