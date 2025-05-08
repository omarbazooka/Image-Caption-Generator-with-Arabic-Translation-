import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer
import nltk
import threading
import time

nltk.download('punkt')


# Load captioning and translation models

caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")  
translation_model_name = "Helsinki-NLP/opus-mt-tc-big-en-ar"  
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)  
model = MarianMTModel.from_pretrained(translation_model_name) 


# Generate English caption

def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')  
    inputs = caption_processor(images=raw_image, return_tensors="pt")  
    out = caption_model.generate(**inputs)  
    caption = caption_processor.decode(out[0], skip_special_tokens=True)  
    return caption  


# Translate to Arabic

def translate_to_arabic(text):
    inputs = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")  
    translated = model.generate(**inputs) 
    return tokenizer.decode(translated[0], skip_special_tokens=True)  


# Main app class

class CaptionApp:
    def __init__(self, root):
        self.root = root  
        self.root.title("🖼️ Image Caption Generator")  
        self.root.geometry("720x650") 
        self.root.configure(bg='#f4f4f4')  

        self.label = tk.Label(root, text="📸 Select an image for a caption and Arabic translation.", font=("Arial", 16), bg='#f4f4f4', fg="#222")
        self.label.pack(pady=15)  

        self.img_label = tk.Label(root, bg='#f4f4f4')
        self.img_label.pack()
        
        style = ttk.Style()  
        style.theme_use("default")  
        style.configure("green.Horizontal.TProgressbar", troughcolor="#e0e0e0", background="#4CAF50", thickness=20, borderwidth=0)  

        self.progress = ttk.Progressbar(root, mode='indeterminate', length=350, style="green.Horizontal.TProgressbar") 

        self.en_caption_label = tk.Label(root, text="English Caption:", font=("Arial", 12), bg='#f4f4f4', fg="#333")
        self.en_caption_label.pack(pady=5)  

        self.ar_caption_label = tk.Label(root, text="Arabic Translation:", font=("Arial", 12), bg='#f4f4f4', fg="#333")
        self.ar_caption_label.pack(pady=5) 

        self.choose_btn = tk.Button(root, text="📂 Choose Image", command=self.choose_image, font=("Arial", 12), bg="#4CAF50", fg="white", width=20)
        self.choose_btn.pack(pady=20)  

        self.error_label = tk.Label(root, text="", font=("Arial", 12), bg='#f4f4f4', fg="red")
        self.error_label.pack(pady=5)  

    
# Choose image and start processing

    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[["Image files", "*.jpg *.jpeg *.png"]]) 
        if file_path:  
            img = Image.open(file_path)  
            img.thumbnail((350, 350))  
            img_tk = ImageTk.PhotoImage(img)  
            self.img_label.config(image=img_tk) 
            self.img_label.image = img_tk 

            self.progress.pack(pady=15) 
            self.progress.start() 
            self.choose_btn.config(state='disabled') 
            self.error_label.config(text="")  
            threading.Thread(target=self.process_image, args=(file_path,)).start() 

    
# Generate caption and translation

    def process_image(self, file_path):
        try:
            time.sleep(2)  
            caption = generate_caption(file_path) 
            translation = translate_to_arabic(caption)  
    
            self.en_caption_label.config(text=f"English Caption: {caption}")
            self.ar_caption_label.config(text=f"Arabic Translation: {translation}")
        except Exception as e:
            
            self.en_caption_label.config(text="❌ Error in captioning")
            self.ar_caption_label.config(text="❌ Error in translation")
            self.error_label.config(text=f"Error: {str(e)}")  
        finally:
            self.progress.stop()  
            self.progress.pack_forget()  
            self.choose_btn.config(state='normal') 


# Run the app

if __name__ == "__main__":  
    root = tk.Tk()  
    app = CaptionApp(root) 
    root.mainloop() 
