

import fitz
file = fitz.open("pictures with text.pdf")
import os, sys
# iterate over PDF pages and find all the images location inside the pdf
import io
from PIL import Image



for i in range(len(file)):
    # get the page itself
    page = file[i]
    #image_list = page.getImageList()
    image_list= page.get_images()
    # printing number of images found in this page
    if image_list:
        print(f"[+] Found a total of {len(image_list)} images in page {i}")
    else:
        print("[!] No images found on page", i)
    for image_index, img in enumerate(page.get_images(), start=1):
        # get the XREF of the image
        xref = img[0]
        # extract the image bytes
        base_image = file.extract_image(xref)
        image_bytes = base_image["image"]
        # get the image extension
        image_ext = base_image["ext"]
        # load it to PIL
        image = Image.open(io.BytesIO(image_bytes))
        # save it to local disk
        image.save(open(f"image{i+1}_{image_index}.{image_ext}", "wb"))
        import os

directory = r'C:\Users\habua\Desktop\Project VSC'
for filename in os.listdir(directory):
    if filename.endswith(".jpeg") or filename.endswith(".png"):
        print(os.path.join(directory, filename))
    else:
        continue
import easyocr
reader= easyocr.Reader(['en'])
#results= reader.readtext('thumbnail_Image-1.png')
#print (results)
import os
path_of_the_directory = directory
ext = ('jfif','png',)
for files in os.listdir(path_of_the_directory):
    if files.endswith(ext):
        print(files)  
        results= reader.readtext(files)
        dialogue= ''
        for result in results:
            dialogue+= result[1]+''
            #print (dialogue)
        print (dialogue)
    else:
        continue