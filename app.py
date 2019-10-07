import os
from flask import Flask,request,render_template,send_from_directory,redirect
from flask_restful import Resource, Api
from flask_jsonpify import jsonify
from Object_detection_image import *
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

app = Flask(__name__)
api = Api(app)
cnt=0
detected=1

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/current_cart'
photos = UploadSet('photos', ('png', 'jpg'))
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

@app.route('/favicon.ico')
def favicon():
	return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def home():
	return render_template('mainLayout.html')

# @app.route('/generate-bill-page', methods=['GET'])
# def BillingPage():
#     return render_template('billingPage.html')

# background process happening without any refreshing
@app.route('/camscan')
def CamScan():
	from os import listdir
	CWD_PATH = os.getcwd()
	img_dir_path=os.path.join(CWD_PATH,"current_cart")

	#empty the contents in the last bill
	open('product_list.csv', 'w').close()

	image_list=listdir(img_dir_path)
	load_images=[]
	for image in image_list:
		global cnt
		global detected
		if cnt==0:
			PATH_TO_IMAGE=os.path.join(img_dir_path,image)

			detected=gvs(PATH_TO_IMAGE)
			cnt+=1
			if detected != 1:
				detected=image
				return redirect('/error')
		else:
			PATH_TO_IMAGE=os.path.join(img_dir_path,image)
			detected=gvs2(PATH_TO_IMAGE)
			if detected!=1:
				detected=image
				return redirect('/error')

	return redirect('/show-bill')

@app.route('/show-bill')
def ShowBill():
	import csv
	from datetime import datetime
	import random
	items_freq={}
	redundant_list=[]
	with open("product_list.csv",'r') as cart:
		csv_reader=csv.reader(cart)
		for line in csv_reader:
			redundant_list.append(int(line[0]))

	# print("redundant_list",redundant_list)

	set_list=list(set(redundant_list))

	# print("set_list",set_list)

	for i in range(len(set_list)):
		items_freq[set_list[i]]=redundant_list.count(set_list[i])

	cart={}
	total_price=0
	for key,value in items_freq.items():
		tempy=product_details[key]
		tempy['qty']=value
		total_price+=tempy['price']
		cart[key]=tempy

	costs={}
	gst_amt=(total_price*5)/100

	costs['gst_amt']=gst_amt
	costs['cost_wo_gst']=total_price
	costs['cost_w_gst']=total_price+gst_amt

	invoice_id=random.randint(10000, 99999)
	timestamp=datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
	return render_template('showBillPage.html',
		cart=cart,
		costs=costs,
		timestamp=timestamp,
		invoice_id=invoice_id)


@app.route('/generate-bill-page', methods=['GET','POST'])
def BillingPage():
	if request.method == 'POST':
		#empty current_cart folder
		import glob
		files = glob.glob(os.getcwd() + '/current_cart/*')
		for f in files:
			os.remove(f)
		#empty current_cart folder
		file_obj = request.files.getlist('file[]')
		for f in file_obj:
			# file = request.files.get(f)
			file = f
			filename = photos.save(file,name=file.filename)
		return redirect('camscan')
	else:
		return render_template('billingPage.html')


@app.route('/error')
def Error():
	global detected
	return render_template('errorPage.html', errorImage=detected)

@app.route('/gvs')
def Gvs():
	return "Fuck world!!"

# api.add_resource(Employees, '/employees') # Route_1
# api.add_resource(Gvs, '/gvs') # Route_2
# api.add_resource(BillingPage, '/billingpage')

if __name__ == '__main__':
	 app.run(port=5002,debug=True)
