import llama_cpp
import numpy as np

class LLaMA:
    def __init__(self):
        self.model = llama_cpp.Llama(
            model_path="./model/Llama-3.2-3B-Instruct-f16.gguf",
            verbose=False,
            n_gpu_layers=-1,
            n_ctx=8192,
            temperature=0.9,
            max_tokens=3
        )
    def post_processed(self, context):
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "你是一個能精確修正語句的助手，專門改善OCR錯誤。 語言:繁體中文"},
                
                {"role": "user", "content": f"OCR結果:「 一 投資 資產 組 合 不 給 是 # 衝 經 風險 調整 後 的 報酬 率 都 沒 有 明顯高」  注意:請將其修正為流暢的語句，僅輸出結果。 語言:繁體中文"},
                {"role": "assistant","content":'一投資資產組合不論是每月、每季或每年進行再平衡，經風險調整後的報酬率都沒有明顯差異'},
                
                {"role": "user", "content": f"OCR結果:「{context}」  注意:請將其修正為流暢的語句，僅輸出結果。 語言:繁體中文"},
            ]
        )

        return response["choices"][0]["message"]["content"]
    
    
    def read_abstract(self, book_name):
        with open(f"./Abstract/{book_name}.txt", "r", encoding="utf-8") as book:
            abstract = book.read()
                 
        return abstract
            
    def read_reason(self, book_content):
        response = self.model.create_chat_completion(
            messages=[
                {"role": "system", "content": "你是一個專業的圖書推薦專家，根據書籍名與摘要給予簡短的推薦理由。 輸出: { 理由(大約80字) } 語言:繁體中文 注意:請全部使用繁體中文進行輸出"},
                
                {"role": "user", "content": f'''
                 Book: 三隻小豬
                 Abstract: 「《三隻小豬》是一個經典的童話故事，講述三隻小豬建造房子的冒險。第一隻小豬用稻草建屋，第二隻用木頭，第三隻則用磚頭。當大野狼來襲時，他用力吹毀了稻草和木頭房子，但無法摧毀堅固的磚頭房子。最終，三隻小豬成功抵擋了狼的攻擊，學會了努力與智慧的重要性。」
                 '''},
                {"role": "assistant","content":'《三隻小豬》是一個寓教於樂的故事，透過輕鬆有趣的情節教導努力、智慧與團結的重要性，非常適合親子共讀，啟發孩子珍惜成果並學會應對挑戰。'},
                                
                {"role": "user", "content": book_content},
            ]
        )
        
        result = response["choices"][0]["message"]["content"]       
        
        
        return result

              
    def recommendation(self, top_books):
        ranks = ""
        for rank, (_, title) in enumerate(top_books, start=1):
            abstract = self.read_abstract(title)
            reason = self.read_reason(f"Book: {title} \nAbstract: 「{abstract})」")
            ranks += f"\nRank {rank} Book: {title} \n推薦原因: {reason}\n"
        
        
        return ranks




if __name__ == "__main__":
    test_context = '''
內 過 如 全 嫩
高 / 再 平衡 行 動 的 次 數 和 成 本 明顯 增加 。,

譯註 : 這 份 研究 報告 定 療
4 和 大 本 包括 稅 頁 、 從 易 成 本 , 以 及 投入 的 時 間 與 勞力 s)
再 由
    '''
    
    llama = LLaMA()
    # print(llama.post_processed(test_context))
    
    
    print(llama.recommendation([(np.float32(13.758684), '富爸爸窮爸爸')]))
