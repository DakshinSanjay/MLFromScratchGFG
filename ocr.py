import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import easyocr
import os

model_folder = os.path.dirname(os.path.abspath(__file__))

# Create main window
root = tk.Tk()
root.title("Simple OCR Tool")
root.geometry("900x500")

# OCR Reader
reader = easyocr.Reader(['en'], model_storage_directory=model_folder, gpu=False)

# Frames for two columns
left_frame = tk.Frame(root, width=450, height=450, bg="white")
left_frame.grid(row=0, column=0, padx=10, pady=10)

right_frame = tk.Frame(root, width=450, height=450, bg="white")
right_frame.grid(row=0, column=1, padx=10, pady=10)

# Canvas for image area (left side)
image_canvas = tk.Canvas(left_frame, width=400, height=350, bg="lightgrey")
image_canvas.pack(pady=10)

# Text box for OCR output (right side)
text_box = tk.Text(right_frame, width=50, height=20, font=("Arial", 12))
text_box.pack(pady=10)

current_image_path = None
image_obj = None

def select_image():
    global current_image_path, image_obj
    path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    if not path:
        return
    current_image_path = path

    img = Image.open(path)
    img = img.resize((400, 350))
    image_obj = ImageTk.PhotoImage(img)

    image_canvas.delete("all")
    image_canvas.create_image(0, 0, anchor="nw", image=image_obj)

def extract_text():
    if not current_image_path:
        return

    results = reader.readtext(current_image_path)
    text_box.delete("1.0", tk.END)

    for box, text, conf in results:
        text_box.insert(tk.END, text + "\n")

def clear_all():
    global current_image_path, image_obj
    current_image_path = None

    image_canvas.delete("all")
    text_box.delete("1.0", tk.END)

# Buttons for left column
button_frame_left = tk.Frame(left_frame)
button_frame_left.pack()

select_button = tk.Button(button_frame_left, text="Select Image", width=15, command=select_image)
select_button.grid(row=0, column=0, padx=5, pady=5)

extract_button = tk.Button(button_frame_left, text="Extract", width=15, command=extract_text)
extract_button.grid(row=0, column=1, padx=5, pady=5)

# Button for right column
clear_button = tk.Button(right_frame, text="Clear All", width=15, command=clear_all)
clear_button.pack(pady=5)

root.mainloop()
