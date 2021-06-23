from flask import Flask,redirect, url_for,render_template,request
import os
#from index import d_dtcn
from index import *
import webbrowser

secret_key = str(os.urandom(24))

app = Flask(__name__)
app.config['TESTING'] = True
app.config['DEBUG'] = True
app.config['FLASK_ENV'] = 'development'
app.config['SECRET_KEY'] = secret_key
app.config['DEBUG'] = True

# Defining the home page of our site
@app.route("/",methods=['GET', 'POST'])
def home():
    print(request.method)
    if request.method == 'POST':
        if request.form.get('Continue') == 'Continue':
           return render_template("test1.html")
    else:
        # pass # unknown
        return render_template("index.html")

@app.route("/start", methods=['GET', 'POST'])
def index():
    print(request.method)
       
    if request.method == 'POST':
        if request.form.get('Start Webcam') == 'Start Webcam':
            num = request.form.get("phnum")
            print(num)
            #num="9946219100"
            d_dtcn(str(num))
            return render_template("index.html")
    else:
        # pass # unknown
        return render_template("index.html")

@app.route("/TTM", methods=['GET', 'POST'])
def index1():
    print(request.method)   
    if request.method == 'POST':
        if request.form.get('Start TTM') == 'Start TTM':
            ttm_func()
            return render_template("index.html")
    else:
        # pass # unknown
        return render_template("index.html")


@app.route('/contact', methods=['GET', 'POST'])
def cool_form():
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('contact.html')

if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:5011/')
    app.run(port=5011,debug=True, use_reloader=False)
    