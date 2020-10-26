import os
from flask import Flask,request,render_template,send_from_directory,redirect,flash,Markup
from flask_restful import Resource, Api
from flask_jsonpify import jsonify
from Object_detection_image import *
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class

app = Flask(__name__)
api = Api(app)
cnt=0
detected=[1,]
visitors=0
customer_name=''
product_cnt=0
sales=0

# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/current_cart'
photos = UploadSet('photos', ('png', 'jpg'))
configure_uploads(app, photos)
patch_request_class(app)  # set maximum file size, default is 16MB

@app.route('/favicon.ico')
def favicon():
	return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
def Inventory():
	global sales,visitors
	return render_template('InventoryPage.html',
		visitors=visitors,
		sales=sales)

# background process happening without any refreshing
@app.route('/camscan')
def CamScan():
	global product_cnt
	from os import listdir
	CWD_PATH = os.getcwd()
	img_dir_path=os.path.join(CWD_PATH,"current_cart")

	#empty the contents in the last bill
	open('product_list.csv', 'w').close()

	image_list=listdir(img_dir_path)
	not_detected=[]
	for image in image_list:
		global cnt,detected
		if cnt==0:
			PATH_TO_IMAGE=os.path.join(img_dir_path,image)

			detected=load_tensorflow_to_memory(PATH_TO_IMAGE)
			cnt+=1
			if detected[0] != 1:
				detected[0]=image
				not_detected.append(image)
			else:
				product_cnt+=detected[1]

		else:
			PATH_TO_IMAGE=os.path.join(img_dir_path,image)
			detected=perform_product_detection(PATH_TO_IMAGE)
			if detected[0]!=1:
				detected[0]=image
				not_detected.append(image)
			else:
				product_cnt+=detected[1]

	if len(not_detected)!=0:
		not_detected_products=''
		for product in not_detected:
			not_detected_products+=str(product)+","

		flash("Could not detect image: "
		  +not_detected_products
		  +Markup(r"<a href='/show-bill'> click here</a>")
		  +" to generate bill without it or generate new bill with new images")
		return redirect('/generate-bill-page')

	return redirect('/show-bill')


@app.route('/show-bill')
def ShowBill():
	import csv
	from datetime import datetime
	import random
	items_freq={}
	redundant_list=[]
	global visitors
	visitors+=1
	with open("product_list.csv",'r') as cart:
		csv_reader=csv.reader(cart)
		for line in csv_reader:
			redundant_list.append(int(line[0]))

	set_list=list(set(redundant_list))

	for i in range(len(set_list)):
		items_freq[set_list[i]]=redundant_list.count(set_list[i])

	cart={}
	total_price=0
	for key,value in items_freq.items():
		tempy=product_details[key]
		tempy['qty']=value
		total_price+=tempy['price']*value
		cart[key]=tempy

	costs={}
	gst_amt=(total_price*5)/100

	costs['gst_amt']=gst_amt
	costs['cost_wo_gst']=total_price
	costs['cost_w_gst']=total_price+gst_amt

	invoice_id=random.randint(10000, 99999)
	timestamp=datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
	global customer_name
	return render_template('showBillPage.html',
		cart=cart,
		costs=costs,
		timestamp=timestamp,
		invoice_id=invoice_id,
		customer_name=customer_name)


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
		global customer_name
		customer_name=request.form['customer_name']
		for f in file_obj:
			# file = request.files.get(f)
			file = f
			if file.filename is not '':
				filename = photos.save(file,name=file.filename)
			else:
				flash('Select/upload atleast 1 product image to continue.')
				return render_template('billingPage.html')
		return redirect('camscan')
	else:
		return render_template('billingPage.html')

@app.route('/profile')
def profile():
	if request.method =="POST":
		return render_template('profile.html')
	else:
		return redirect('/')

@app.route('/update-sales')
def UpdateSales():
	global product_cnt,sales
	sales=product_cnt
	return redirect('/')

@app.route('/statement', methods=["GET", "POST"])
def statement():
    if request.method == 'POST':
        return render_template("statement.html")
    else:
        ssn = session['accssn']
        acc_id = session['accssn']
        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM transactions WHERE ssn=%s ORDER BY trdate DESC LIMIT 10",(ssn,))
        result = curl.fetchall()
        # print(result)
        curl.close()
        return render_template("statement.html",statements=result)

if __name__ == '__main__':
	app.secret_key = 'super secret key'
	app.config['SESSION_TYPE'] = 'filesystem'
	app.run(port=5002,debug=True,host='0.0.0.0')
