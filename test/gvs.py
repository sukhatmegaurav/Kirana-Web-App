from PIL import Image 
import os 
directory = r'B:\OBJDectExp\webflask\test' 
for filename in os.listdir(directory): 
    if filename.endswith(".webp"): 
        prefix = filename.split(".webp")[0]
        im = Image.open(filename)
        im.save(prefix+'.jpg')
    else: 
        continue