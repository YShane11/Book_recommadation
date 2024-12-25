from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r".\Tesseract-OCR\tesseract.exe"


class OCR:
    def adaptive_enhance(self, image):
        gray_image = image.convert("L")
        gray_array = np.array(gray_image)

        mean_brightness = gray_array.mean()  
        std_contrast = gray_array.std()     

        if mean_brightness < 100:  
            brightness_factor = 1.2
        elif mean_brightness > 200: 
            brightness_factor = 0.8
        else:
            brightness_factor = 1.0 

        if std_contrast < 50: 
            contrast_factor = 1.5
        elif std_contrast > 100: 
            contrast_factor = 0.8
        else:
            contrast_factor = 1.0 

        image = ImageEnhance.Brightness(image).enhance(brightness_factor)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        return image
    
    def scan(self, filepath):
        image = Image.open(filepath)
        
        image = self.adaptive_enhance(image).convert('L')
        
        # image = image.point(lambda x: 0 if x < 150 else 255, '1')
        
        image = image.filter(ImageFilter.SHARPEN)
               
        return pytesseract.image_to_string(image, lang='chi_tra', config='--psm 1 --oem 1')

if __name__ == "__main__":
    ocr = OCR()
    # print(ocr.scan("./test.png"))
    print(ocr.scan(r"static\captures\test_1.jpg"))

