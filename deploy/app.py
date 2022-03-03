from flask import Flask, render_template , request
import predict as model
import torchvision.transforms as T
from torchvision.utils import save_image
from PIL import Image
import os

IMG_FOLDER = os.path.join('static', 'images')
RES_FOLDER = os.path.join('static', 'results')

app = Flask( __name__)
app.config['UPLOAD_FOLDER'] = IMG_FOLDER
app.config['DISPLAY_FOLDER'] = RES_FOLDER

@app.route('/',methods = ['GET'])
def hello_world():
    return render_template("index.html")

@app.route('/',methods = ['POST'])
def predict():
    imagefile = request.files['img_file']
    image_path = "./static/images/" + imagefile.filename
    imagefile.save(image_path)

    config1  = model.CycleGANConfig()
    config1.weights_gen_AB = "Gen_A_to_B.pth"
    config1.weights_gen_BA = "Gen_B_to_A.pth"
    config1.weights_dis_A = "Dis_A.pth"
    config1.weights_dis_B = "Dis_B.pth"
    config1.input_dims = (1,512,512)

    img = Image.open(image_path)
    
    md = model.CycleGAN(config1)
    mri = md.predict(img)

    img1 = mri[0]
    result_path =  "./static/results/"+ "mri_"+ imagefile.filename
    save_image(img1 , result_path)
    result_os_path = os.path.join(app.config['DISPLAY_FOLDER'],  "mri_"+ imagefile.filename)
    input_os_path = os.path.join(app.config['UPLOAD_FOLDER'],imagefile.filename)
    
    return render_template("index.html" ,input_path = input_os_path , result_path = result_os_path)

if __name__ == '__main__':
    app.run(port=3000,debug=True)
