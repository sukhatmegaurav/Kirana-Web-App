import os
from flask import Flask,request,render_template,send_from_directory,redirect
from flask_restful import Resource, Api
from flask_jsonpify import jsonify
from Object_detection_image import *

app = Flask(__name__)
api = Api(app)
cnt=0

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def home():
    return render_template('mainLayout.html')

@app.route('/generate-bill-page')
def BillingPage():
    return render_template('billingPage.html') 

# background process happening without any refreshing
@app.route('/camscan')
def CamScan():
	from os import listdir
	CWD_PATH = os.getcwd()
	img_dir_path=os.path.join(CWD_PATH,"test")

	#empty the contents in the last bill
	open('product_list.csv', 'w').close()
	
	image_list=listdir(img_dir_path)
	load_images=[]
	for image in image_list:
		global cnt
		if cnt==0:
			PATH_TO_IMAGE=os.path.join(img_dir_path,image)
			gvs(PATH_TO_IMAGE)
			cnt+=1
		else:
			PATH_TO_IMAGE=os.path.join(img_dir_path,image)
			gvs2(PATH_TO_IMAGE)
	
	return redirect('/show-bill')

@app.route('/show-bill')
def ShowBill():
	
    return render_template('showBillPage.html') 

@app.route('/gvs')
def Gvs():
    return "Fuck world!!" 

# api.add_resource(Employees, '/employees') # Route_1
# api.add_resource(Gvs, '/gvs') # Route_2
# api.add_resource(BillingPage, '/billingpage')

if __name__ == '__main__':
     app.run(port=5002,debug=True)