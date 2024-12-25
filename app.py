from flask import Flask, render_template, request, jsonify
import os
import base64
import llama_cpp
from datetime import datetime
from utils.ocr import OCR
from utils.LLaMA import LLaMA
from utils.rag import VectorHandler
import time


app = Flask(__name__)
llama = LLaMA()
vectorhandle = VectorHandler()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["POST"])
def capture():
    start_time = time.time() 
    data = request.json
    input_type = data.get("type")
    
    if input_type == "text":
        text_input = data.get("image")  # 這裡的 image 將作為文字輸入

        print(f'Text input: {text_input}')
        ocr_postprocessed = text_input

    elif input_type in ["camera", "upload"]:
        image_data = data.get("image")
        if not image_data:
            return jsonify({"message": "未提供圖像內容"}), 400

        # 根據當前時間生成檔名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 格式: YYYYMMDD_HHMMSS
        filename = f"captured_{current_time}.png"
        image_data = base64.b64decode(image_data.split(",")[-1])
        image_path = os.path.join("static/captures", filename)
        os.makedirs(os.path.dirname(image_path), exist_ok=True)  # 確保目錄存在
        with open(image_path, "wb") as f:
            f.write(image_data)

        # OCR 處理
        ocr = OCR()
        ocr_get = ocr.scan(image_path)
        print(f'ocr_get: {ocr_get}')
        if not ocr_get:
            return jsonify({"message": "未讀取到文字內容"}), 400
        
        ocr_postprocessed = llama.post_processed(ocr_get)

    else:
        return jsonify({"message": "未知的輸入類型"}), 400

    print(f'ocr_postprocessed: {ocr_postprocessed}')
    
    # 書籍推薦處理
    top_books = vectorhandle.recommending_book(ocr_postprocessed)
    print(f'top_books: {top_books}')
    
    recommendation = llama.recommendation(top_books)
    print(f'recommendation: {recommendation}')

    response_data = {"message": recommendation.replace("\n", "<br>")}
    if input_type in ["camera", "upload"]:
        response_data["image_path"] = image_path
        
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Execution time: {execution_time:.2f} seconds')
    
    
    return jsonify(response_data), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
