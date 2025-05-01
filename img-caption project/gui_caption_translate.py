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

# تحميل الأدوات اللازمة من مكتبة nltk
nltk.download('punkt')

# تحميل موديلات BLIP و Helsinki الخاصة بالترجمة
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  # تحميل معالج BLIP
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")  # تحميل موديل BLIP
translation_model_name = "Helsinki-NLP/opus-mt-tc-big-en-ar"  # تعيين موديل الترجمة من الإنجليزية إلى العربية
tokenizer = MarianTokenizer.from_pretrained(translation_model_name)  # تحميل tokenizer الخاص بالترجمة
model = MarianMTModel.from_pretrained(translation_model_name)  # تحميل موديل الترجمة

# دالة لتوليد الكابتشن (الوصف) من الصورة
def generate_caption(image_path):
    raw_image = Image.open(image_path).convert('RGB')  # فتح الصورة وتحويلها إلى تنسيق RGB
    inputs = caption_processor(images=raw_image, return_tensors="pt")  # تجهيز الصورة لموديل BLIP
    out = caption_model.generate(**inputs)  # توليد الوصف باستخدام الموديل
    caption = caption_processor.decode(out[0], skip_special_tokens=True)  # تحويل النتيجة إلى نص
    return caption  # إرجاع النص الناتج

# دالة لترجمة النصوص من الإنجليزية إلى العربية
def translate_to_arabic(text):
    inputs = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt")  # تجهيز النص للترجمة
    translated = model.generate(**inputs)  # إجراء الترجمة باستخدام موديل MarianMT
    return tokenizer.decode(translated[0], skip_special_tokens=True)  # إرجاع الترجمة بعد فك التشفير

# تعريف فئة التطبيق التي تحتوي على الواجهة الرسومية
class CaptionApp:
    def __init__(self, root):
        self.root = root  # تخزين نافذة التطبيق
        self.root.title("🖼️ Image Caption Generator")  # تعيين عنوان النافذة
        self.root.geometry("720x650")  # تعيين أبعاد النافذة
        self.root.configure(bg='#f4f4f4')  # تعيين لون خلفية النافذة

        # إضافة تسمية لشرح المطلوب من المستخدم
        self.label = tk.Label(root, text="📸 Select an image for a caption and Arabic translation.", font=("Arial", 16), bg='#f4f4f4', fg="#222")
        self.label.pack(pady=15)  # إضافة التسمية في النافذة مع مسافة بادئة

        # إضافة مكان لعرض الصورة التي يختارها المستخدم
        self.img_label = tk.Label(root, bg='#f4f4f4')
        self.img_label.pack()

        # تصميم ستايل مخصص للـ progress bar (شريط التقدم)
        style = ttk.Style()  # إنشاء ستايل جديد
        style.theme_use("default")  # استخدام الثيم الافتراضي
        style.configure("green.Horizontal.TProgressbar", troughcolor="#e0e0e0", background="#4CAF50", thickness=20, borderwidth=0)  # تخصيص مظهر شريط التقدم

        # إضافة شريط التقدم
        self.progress = ttk.Progressbar(root, mode='indeterminate', length=350, style="green.Horizontal.TProgressbar")  # شريط التقدم بنمط غير محدد

        # إضافة تسمية لعرض الوصف باللغة الإنجليزية
        self.en_caption_label = tk.Label(root, text="English Caption:", font=("Arial", 12), bg='#f4f4f4', fg="#333")
        self.en_caption_label.pack(pady=5)  # إضافة التسمية في النافذة مع مسافة بادئة

        # إضافة تسمية لعرض الترجمة بالعربية
        self.ar_caption_label = tk.Label(root, text="Arabic Translation:", font=("Arial", 12), bg='#f4f4f4', fg="#333")
        self.ar_caption_label.pack(pady=5)  # إضافة التسمية في النافذة مع مسافة بادئة

        # إضافة زر لاختيار الصورة من النظام
        self.choose_btn = tk.Button(root, text="📂 Choose Image", command=self.choose_image, font=("Arial", 12), bg="#4CAF50", fg="white", width=20)
        self.choose_btn.pack(pady=20)  # إضافة الزر في النافذة مع مسافة بادئة

        # إضافة مكان لعرض رسائل الخطأ
        self.error_label = tk.Label(root, text="", font=("Arial", 12), bg='#f4f4f4', fg="red")
        self.error_label.pack(pady=5)  # إضافة التسمية في النافذة مع مسافة بادئة

    # دالة لاختيار الصورة من جهاز الكمبيوتر
    def choose_image(self):
        file_path = filedialog.askopenfilename(filetypes=[["Image files", "*.jpg *.jpeg *.png"]])  # فتح نافذة لاختيار ملف صورة
        if file_path:  # إذا تم اختيار صورة
            img = Image.open(file_path)  # فتح الصورة
            img.thumbnail((350, 350))  # تصغير الصورة
            img_tk = ImageTk.PhotoImage(img)  # تحويل الصورة إلى تنسيق مناسب لـ tkinter
            self.img_label.config(image=img_tk)  # تعيين الصورة في التسمية الخاصة بعرض الصورة
            self.img_label.image = img_tk  # تخزين الصورة لكي لا يتم حذفها

            self.progress.pack(pady=15)  # إظهار شريط التقدم
            self.progress.start()  # بدء شريط التقدم
            self.choose_btn.config(state='disabled')  # تعطيل زر اختيار الصورة أثناء المعالجة
            self.error_label.config(text="")  # مسح أي رسائل خطأ سابقة
            threading.Thread(target=self.process_image, args=(file_path,)).start()  # تشغيل دالة المعالجة في خيط منفصل

    # دالة لمعالجة الصورة: توليد الوصف وترجمته
    def process_image(self, file_path):
        try:
            time.sleep(2)  # إضافة تأخير لمحاكاة المعالجة
            caption = generate_caption(file_path)  # توليد الوصف للصورة
            translation = translate_to_arabic(caption)  # ترجمة الوصف إلى العربية

            # تحديث التسمية لعرض الوصف باللغة الإنجليزية والترجمة بالعربية
            self.en_caption_label.config(text=f"English Caption: {caption}")
            self.ar_caption_label.config(text=f"Arabic Translation: {translation}")
        except Exception as e:
            # في حالة حدوث خطأ، عرض رسالة الخطأ في الواجهة
            self.en_caption_label.config(text="❌ Error in captioning")
            self.ar_caption_label.config(text="❌ Error in translation")
            self.error_label.config(text=f"Error: {str(e)}")  # عرض رسالة الخطأ
        finally:
            self.progress.stop()  # إيقاف شريط التقدم
            self.progress.pack_forget()  # إخفاء شريط التقدم
            self.choose_btn.config(state='normal')  # إعادة تمكين زر اختيار الصورة

# تشغيل التطبيق
if __name__ == "__main__":  # إذا كان هذا هو السكربت الرئيسي الذي يتم تشغيله
    root = tk.Tk()  # إنشاء نافذة tkinter جديدة
    app = CaptionApp(root)  # إنشاء مثيل من فئة التطبيق
    root.mainloop()  # تشغيل التطبيق وإظهار النافذة
