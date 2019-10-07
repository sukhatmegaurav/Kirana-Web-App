# from PIL import Image 
# import os 
# directory = r'B:\OBJDectExp\webflask\test' 
# for filename in os.listdir(directory): 
#     if filename.endswith("webp"): 
#         prefix = filename.split("webp")[0]
#         im = Image.open(filename).convert("RGB")
#         im.save(prefix+'.jpg',"jpeg")  
#     else: 
#         continue
from PIL import Image
import os

for (dirname, dirs, files) in os.walk("."):
     for filename in files:
         if filename.endswith('.webp'):
             print('found: ' + os.path.splitext(filename)[0])
             print('converting to: ' + os.path.splitext(filename)[0] + '.jpg')
             im = Image.open(filename).convert("RGB")
             im.save(os.path.splitext(filename)[0] + '.jpg', "jpeg")
             print('done converting…')