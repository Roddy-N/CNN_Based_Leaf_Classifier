import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from tkinter import ttk
import tensorflow as tf
import numpy as np
import cv2
from fun import class_list

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master,bg='gray')
        self.master = master
        self.master.title("Leaf Classifier")
        self.pack(fill="both", expand=True)
        self.create_widgets()
        self.load_model()
        self.adjust_window_size()
        self.show_converted = False

    def adjust_window_size(self):
        width = self.master.winfo_screenwidth()
        height = self.master.winfo_screenheight()
        self.master.geometry(f"{width}x{height}")

    def create_widgets(self):
        # create left frame
        self.left_frame = tk.Frame(self, bg='gray')
        self.left_frame.pack(side="left", padx=5, pady=5, fill="both", expand=True)

        # create select image button
        self.browse_button = tk.Button(self.left_frame, text="Select Image", command=self.select_image)
        self.browse_button.pack(side="bottom", padx=10, pady=5, anchor='w')

        self.image_canvas = tk.Canvas(self.left_frame, width=550, height=700)
        self.image_canvas.pack(side="left", padx=10, pady=10, anchor='center')

        self.clear_button = tk.Button(self.left_frame, text="Clear", command=self.clear_image)
        self.clear_button.pack(side="bottom", padx=10, pady=10, anchor='e')

        # create predict image label
        self.predict_label = tk.Label(self.left_frame, text="Predict Image", bg="gray", fg='white', font=("Arial", 24,'bold'))
        self.predict_label.pack(side="top", padx=10, pady=10, anchor='n')

        # create right frame
        self.right_frame = tk.Frame(self, bg='gray')
        self.right_frame.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        self.species_label = tk.Label(self.right_frame, text="Species of plants", bg='gray', fg='white', font=('Arial', 24, 'bold'))
        self.species_label.pack(side='top', padx=10, pady=10, anchor='nw')

        self.listbox = tk.Listbox(self.right_frame, width=30, height=5, bg="gray")
        self.listbox.pack(side="left", padx=10, pady=10, fill="both", expand=True)

        for word in class_list():
            self.listbox.insert(tk.END, word)

        scrollbar = tk.Scrollbar(self.listbox)
        scrollbar.pack(side="right", fill="y")
        self.listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.listbox.yview)

        # create bottom buttons and result label
        self.convert_button = tk.Button(self.left_frame, text="Predict", command=self.toggle_conversion)
        self.convert_button.pack(side="left", padx=10, pady=10, anchor='w')

        self.result_label = tk.Label(self.left_frame, text="", width=40, height=5)
        self.result_label.pack(side="left", padx=10, pady=10, anchor='center')


    def load_model(self):
        # Load pre-trained CNN model
        self.model = tf.keras.models.load_model("/home/wolf/Desktop/TK tutorial/leaf.h5")

    def select_image(self):
        file_path = filedialog.askopenfilename()#title="Select Image", filetypes=filetypes)
        if file_path:
            try:
                # Load image
                image = Image.open(file_path)
                self.original_image = image.copy()

                # Display original image
                self.photo = ImageTk.PhotoImage(image)
                self.image_canvas.create_image(0, 0, image=self.photo, anchor="nw")

            except Exception as e:
                messagebox.showerror("Error", str(e))

    def clear_image(self):
        self.image_canvas.delete("all")
        self.result_label.configure(text="")


    def toggle_conversion(self):
            if not self.show_converted:
                try:
                    # Convert image to grayscale
                    gray = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_BGR2GRAY)

                    # Detect edges using Canny algorithm
                    edges = cv2.Canny(gray, 50, 150)

                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    external_edges = np.zeros_like(edges)
                    cv2.drawContours(external_edges, contours, -1, (255, 255, 255), 1)

                    # Display edges image
                    self.converted_photo = ImageTk.PhotoImage(Image.fromarray(external_edges))
                    self.image_canvas.create_image(0, 0, image=self.converted_photo, anchor="nw")

                    # Process edges image with CNN model
                    img = np.array(cv2.resize(external_edges, (256, 256))) / 255.0  # Resize and normalize image
                    img = np.expand_dims(img, axis=-1)  # Add channel dimension
                    img = np.expand_dims(img, axis=0)  # Add batch dimension

                    # Make predictions on the input image
                    image = Image.fromarray(gray)
                    image = image.resize((256, 256))
                    image = image.convert('L')
                    input_arr = np.array(image)
                    input_arr = np.array([input_arr])
                    predictions = self.model.predict(input_arr)

                    # Display results
                    class_names = class_list() # Replace with your own class names
                    max_idx = np.argmax(predictions[0])
                    class_name = class_names[max_idx]
                    percentage = round(predictions[0][max_idx]*100, 2)
                    result_text = f"{class_name}"
                    self.result_label.configure(text=result_text)

                except Exception as e:
                    messagebox.showerror("Error", str(e))


    def clear_image(self):
        self.image_canvas.delete("all")
        self.result_label.configure(text="")

root = tk.Tk()
app = Application(master=root)
app.mainloop()
