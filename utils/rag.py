from text2vec import SentenceModel
from pydantic import BaseModel
from tqdm import tqdm
import numpy as np
import os

class BookContext(BaseModel):
    title: str
    context: list

class VectorHandler:
    def __init__(self, embedding_model_name="shibing624/text2vec-base-chinese"):
        self.embedding = SentenceModel(embedding_model_name)

    def encoder(self, text):
        return np.array(self.embedding.encode(text))

    def text_split(self, title, context):
        return BookContext(title=title, context=context.split("\n\n"))

    def save_to_file(self, vector, filename):
        path = f"./vector/{filename}.npz"
        np.savez(file=path, vector=vector)
        return path

    def read_from_file(self, vector_path):
        file_name = vector_path.split("/")[-1]
        _vector = np.load(vector_path)["vector"]

        return file_name, _vector

    def calculate_similarity(self, query_vector, books_vectors):
        return np.min([np.linalg.norm(query_vector - v) for v in books_vectors])

    def recommending_book(self, ocr_result):
        top_n = 3
        distances_and_titles = []
        
        ocr_result_vector = self.encoder(ocr_result)
        for book_vector_path in os.listdir("./vector"):
            _, books_vectors = self.read_from_file(f"./vector/{book_vector_path}")
            vector_distance = self.calculate_similarity(ocr_result_vector, books_vectors)
            distances_and_titles.append((vector_distance, book_vector_path.replace(".npz","")))
            
        top_books = sorted(distances_and_titles, key=lambda x: x[0])[:top_n]
        return top_books
    
if __name__ == "__main__":
    # book_name = "鳥人計畫"
    # with open(f"./books/{book_name}.txt", "r", encoding="utf-8") as book:
    #     txt = book.read()

    # vector_handler = VectorHandler()
    # book_context = vector_handler.text_split(title=book_name, context=txt)

    # vectors = [vector_handler.encoder(i) for i in tqdm(book_context.context)]
    # vector_handler.save_to_file(vectors, book_name)
    
    
    vector_handler = VectorHandler()
    ocr_result = '''
    我在年青時候也曾經有過許多夢
    '''
    
    
    top_books = vector_handler.recommending_book(ocr_result)
    
    # print(top_books)
    
    for rank, (distance, title) in enumerate(top_books, start=1):
        print(f"Rank {rank}: {title} with distance {distance}")
