import tkinter as tk
from tkinter import filedialog
import csv
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageTk


class BodyPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Body Prediction App")

        # สร้าง Frame สำหรับแสดงข้อมูล
        self.data_frame = tk.Frame(root)
        self.data_frame.pack(pady=20)

        # สร้าง Label สำหรับหัวข้อข้อมูล
        self.header_label = tk.Label(self.data_frame, text="SEX\tAGE\tWEIGHT\tHEIGHT\tBody Type")
        self.header_label.grid(row=0, column=0, columnspan=2, padx=5)

        # สร้าง Text widget สำหรับแสดงข้อมูล
        self.data_text = tk.Text(self.data_frame, height=10, width=50)
        self.data_text.grid(row=1, column=0, columnspan=2, padx=5)

        # สร้าง Text widget สำหรับแสดงผลลัพธ์
        self.result_text = tk.Text(root, height=1, width=20)
        self.result_text.pack(pady=10)

        # ช่องใส่ข้อความ
        self.textbox_frame = tk.Frame(root)
        self.textbox_frame.pack(side=tk.RIGHT, padx=20)

        self.sex_label = tk.Label(self.textbox_frame, text="SEX:")
        self.sex_label.grid(row=0, column=0, padx=5, pady=5)
        self.sex_entry = tk.Entry(self.textbox_frame)
        self.sex_entry.grid(row=0, column=1, padx=5, pady=5)

        self.age_label = tk.Label(self.textbox_frame, text="AGE:")
        self.age_label.grid(row=1, column=0, padx=5, pady=5)
        self.age_entry = tk.Entry(self.textbox_frame)
        self.age_entry.grid(row=1, column=1, padx=5, pady=5)

        self.weight_label = tk.Label(self.textbox_frame, text="WEIGHT:")
        self.weight_label.grid(row=2, column=0, padx=5, pady=5)
        self.weight_entry = tk.Entry(self.textbox_frame)
        self.weight_entry.grid(row=2, column=1, padx=5, pady=5)

        self.height_label = tk.Label(self.textbox_frame, text="HEIGHT:")
        self.height_label.grid(row=3, column=0, padx=5, pady=5)
        self.height_entry = tk.Entry(self.textbox_frame)
        self.height_entry.grid(row=3, column=1, padx=5, pady=5)

        # ปุ่มเปิดไฟล์ภาพ
        self.open_image_button = tk.Button(root, text="Open Image", command=self.open_image)
        self.open_image_button.pack(pady=10)

        # ปุ่มเปิดไฟล์ CSV
        self.open_csv_button = tk.Button(root, text="Open CSV", command=self.open_csv)
        self.open_csv_button.pack(pady=10)

        # KNeighborsClassifier
        self.knn_classifier = KNeighborsClassifier(n_neighbors=3)

         # เรียกเมื่อสร้าง instance ใหม่เพื่อแสดงปุ่ม "Predict"
        self.reset_prediction()

        # ปุ่มกด "Predict"
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_new_data)
        self.predict_button.pack(pady=20)

        # ปุ่ม "Predict New Data"
        self.predict_new_button = tk.Button(root, text="Predict New Data", command=self.predict_new_data)
        self.predict_new_button.pack_forget()  # ซ่อนปุ่มเริ่มต้น

        


    # ปุ่มกด "Predict"
    def predict_new_data(self):
        # ดึงข้อมูลจากช่องใส่ข้อความ
        sex = float(self.sex_entry.get())
        age = float(self.age_entry.get())
        weight = float(self.weight_entry.get())
        height = float(self.height_entry.get())

        # ใส่ข้อมูลในโมเดล KNeighborsClassifier
        X = [[sex, age, weight, height]]
        

        # ตรวจสอบจำนวนตัวอย่างที่ใช้ในการทำนาย
        if len(X[0]) != self.knn_classifier.n_features_in_:
            print("Error: Number of features for prediction is not consistent.")
            return

        # ทำนาย
        prediction = self.knn_classifier.predict(X)
        result = f"Body Type Prediction: {prediction[0]}"
        self.result_text.delete(1.0, tk.END)  # ลบข้อมูลเก่าใน Text widget
        self.result_text.insert(tk.END, result)  # เพิ่มข้อมูลทำนายใหม่
    
        # ลบข้อมูลเก่าใน Text widget
        self.result_text.delete(1.0, tk.END)

        # แสดงผลลัพธ์
        result = f"Body Type Prediction: {prediction[0]}"
        self.result_text.insert(tk.END, result)

        # ซ่อนปุ่ม "Predict" และแสดงปุ่ม "Predict New Data"
        self.predict_button.pack_forget()
        self.predict_new_button.pack(pady=20)

        # เรียกเพื่อล้างผลลัพธ์และแสดงปุ่ม "Predict"
        self.reset_prediction()

    def reset_prediction(self):
        # ลบข้อมูลเก่าใน Text widget
        self.result_text.delete(1.0, tk.END)
    
        # แสดงปุ่ม "Predict" และซ่อนปุ่ม "Predict New Data"
        self.predict_button.pack(pady=20)
        self.predict_new_button.pack_forget()
       

    def open_image(self):
        # เปิดไฟล์ภาพ
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            # แสดงภาพในหน้าต่างใหม่
            image = tk.PhotoImage(file=file_path)
            tk.Label(self.root, image=image).pack()
            self.root.mainloop()

    def open_csv(self):
        # เปิดไฟล์ CSV
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            # แสดงข้อมูล CSV ในหน้าต่างใหม่
            with open(file_path, 'r') as file:
                csv_reader = csv.DictReader(file)
                self.data_text.delete(1.0, tk.END)  # ลบข้อมูลเก่าใน Text widget
                self.data_text.insert(tk.END, "SEX\tAGE\tWEIGHT\tHEIGHT\tBody Type\n")  # เพิ่มหัวข้อข้อมูล
                for row in csv_reader:
                    sex, age, weight, height, body_type = map(float, (row['SEX'], row['AGE'], row['WEIGHT'], row['HEIGHT'], self.map_body_type(row['BODY'])))
                    data_row = f"{sex}\t{age}\t{weight}\t{height}\t{body_type}\n"
                    self.data_text.insert(tk.END, data_row)
    def train_model(self):
        # เปิดไฟล์ CSV
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            # อ่านข้อมูล CSV และฝึกสอนโมเดล
            with open(file_path, 'r') as file:
                csv_reader = csv.DictReader(file)
                X, y = [], []
                for row in csv_reader:
                    sex, age, weight, height, body_type = map(float, (row['SEX'], row['AGE'], row['WEIGHT'], row['HEIGHT'], self.map_body_type(row['BODY'])))
                    X.append([sex, age, weight, height])
                    y.append(body_type)
                self.knn_classifier.fit(X, y)

    def map_body_type(self, body_type):
        # แปลง 'thin', 'fat', 'slim' เป็นตำแหน่งเลข
        mapping = {'thin': 1, 'fat': 2, 'slim': 3}
        return mapping.get(body_type, 0)
    
if __name__ == "__main__":
    root = tk.Tk()
    app = BodyPredictionApp(root)
    app.train_model()  # เรียกฟังก์ชันฝึกสอนโมเดล
    root.mainloop()
