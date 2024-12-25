from PIL import Image, ImageEnhance
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r".\Tesseract-OCR\tesseract.exe"

class OCR:
    def scan(self, filepath):
        image = Image.open(filepath)
        enhancer = ImageEnhance.Contrast(image) # 增強對比
        
        image = ImageEnhance.Brightness(image).enhance(1.2)
        image = ImageEnhance.Contrast(image).enhance(2.5)

        
        image.show()
        return pytesseract.image_to_string(image, lang='chi_tra', config='--psm 3 --oem 1')

if __name__ == "__main__":
    ocr = OCR()
    # print(ocr.scan("./test.png"))
    print(ocr.scan(r"static\captures\capture_20241213T204559.png"))

