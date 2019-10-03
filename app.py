import os
from flask import Flask, request,render_template,send_from_directory
from flask_restful import Resource, Api
from flask_jsonpify import jsonify
from Object_detection_image import *

app = Flask(__name__)
api = Api(app)

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def home():
    return render_template('mainLayout.html')

@app.route('/billingpage')
def BillingPage():
    return render_template('billingPage.html') 

# background process happening without any refreshing
@app.route('/camscan')
def CamScan():
    gvs()
    return "nothing"

@app.route('/employees')
def Employees():
    return {'employees': [{'id':1, 'name':'Balram'},{'id':2, 'name':'Tom'}]} 

@app.route('/gvs')
def Gvs():
    return "Fuck world!!" 

# api.add_resource(Employees, '/employees') # Route_1
# api.add_resource(Gvs, '/gvs') # Route_2
# api.add_resource(BillingPage, '/billingpage')

if __name__ == '__main__':
     app.run(port=5002,debug=True)