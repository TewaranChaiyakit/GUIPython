# นำเข้าไลบรารี
import tkinter as tk
from tkinter import filedialog, ttk
from ttkthemes import ThemedTk
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image, ImageTk

# สร้างหน้าต่างหลักของ Tkinter และกำหนดค่าเริ่มต้น
root = ThemedTk(theme="smog")
root.title("RandomForest Predictor")
root.geometry("1080x720")
root.configure(bg="#3292bf")

# ตัวแปร global สำหรับเก็บข้อมูลและหมวดหมู่ต้นฉบับ
global data, original_categories

# ฟังก์ชันสำหรับเรียกหน้าต่างเลือกไฟล์ CSV และโหลดข้อมูล
def browse_file():
    global data, original_categories
    file_path = filedialog.askopenfilename()
    data = pd.read_csv(file_path)
    original_categories = data['DIABETES'].unique()
    show_data(data)


# ฟังก์ชันสำหรับแสดงข้อมูลบน Treeview
def show_data(data):
    treeview.delete(*treeview.get_children())
    for index, row in data.iterrows():
        values = (row['SEX'], row['AGE'], row['WEIGHT'], row['HEIGHT'], row['SMOKING'], row['DRINK_ALCOHOL'], row['DIABETES'])
        treeview.insert("", "end", values=values)
    for col in columns:
        treeview.heading(col, text=col, anchor='center')
        treeview.column(col, anchor='center')


# ฟังก์ชันสำหรับเรียกหน้าต่างเลือกรูปภาพและแสดงรูป
#def browse_image():
#    file_path = filedialog.askopenfilename()
#    if file_path:
#        img = Image.open(file_path)
#        img = resize_image(img, (300, 300))
#        img = ImageTk.PhotoImage(img)
#        lbl_img.config(image=img)
#        lbl_img.image = img


# ฟังก์ชันสำหรับปรับขนาดรูปภาพ
#def resize_image(img, new_size):
#    return img.resize(new_size, resample=Image.BICUBIC)


# ฟังก์ชันสำหรับการฝึก Random Forest และทำนาย
def train_random_forest():
    global rf_model, original_categories
    original_categories = data['DIABETES'].unique()
    data['DIABETES'] = data['DIABETES'].astype('category')
    data['DIABETES'] = data['DIABETES'].cat.codes

    # เพิ่ม features การสูบบุหรี่และการดื่มแอลกอฮอล์เข้าไปใน X
    X = data[['SEX', 'AGE', 'WEIGHT', 'HEIGHT', 'SMOKING', 'DRINK_ALCOHOL']]

    y = data['DIABETES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    prediction = rf_model.predict([[int(entry_sex.get()), int(entry_age.get()), int(entry_weight.get()),
                                    int(entry_height.get()), int(entry_smoking.get()), int(entry_drinking.get())]])
    predicted_category = original_categories[prediction[0]]
    result_var.set(f'Predicted DIABETES: {predicted_category}')
    show_result_window(predicted_category)
    data['DIABETES'] = pd.Categorical.from_codes(data['DIABETES'], categories=original_categories)


# ฟังก์ชันสำหรับรีเซ็ต Random Forest
def reset_random_forest():
    global rf_model, original_categories
    rf_model = None
    original_categories = None


# ฟังก์ชันสำหรับรีเซ็ตข้อมูลที่ป้อน
def reset_inputs():
    entry_age.delete(0, tk.END)
    entry_weight.delete(0, tk.END)
    entry_height.delete(0, tk.END)
    entry_sex.delete(0, tk.END)
    entry_smoking.delete(0, tk.END)
    entry_drinking.delete(0, tk.END)
    result_var.set("Predicted DIABETES: ")
    reset_random_forest()


# ฟังก์ชันสำหรับรีเซ็ตรูปภาพที่แสดง
#def reset_image():
#    lbl_img.config(image=None)
#    lbl_img.image = None


# ฟังก์ชันสำหรับแสดงหน้าต่างผลลัพธ์
def show_result_window(predicted_category):
    result_window = tk.Toplevel(root)
    result_window.title("Predicted DIABETES Result")

    result_label = tk.Label(result_window, text=f'Predicted DIABETES: {predicted_category}', font=("Arial", 20))
    result_label.pack(pady=20)

    ok_button = tk.Button(result_window, text="OK", command=result_window.destroy, bg="#1E90FF", fg="white")
    ok_button.pack()


# สร้าง Frame สำหรับ Widgets ที่ใช้ในการป้อนข้อมูล
frm_input = tk.Frame(root, padx=10, pady=10)
frm_input.pack()

# สร้าง Canvas สำหรับแสดงข้อมูลบน Treeview
canvas = tk.Canvas(root, height=600, width=500)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# กำหนดคอลัมน์ของ Treeview
columns = ['SEX', 'AGE', 'WEIGHT', 'HEIGHT', 'SMOKING', 'DRINK_ALCOHOL', 'DIABETES']
treeview = ttk.Treeview(canvas, columns=columns, show='headings')

# กำหนดขนาดตัวอักษรสำหรับหัวข้อคอลัมน์ใน Treeview
font_size = 12  # ปรับค่านี้ตามต้องการ
style = ttk.Style()
style.configure("Treeview.Heading", font=(None, font_size))

# กำหนดหัวข้อคอลัมน์ใน Treeview
for col in columns:
    treeview.heading(col, text=col)

# Pack Treeview
treeview.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# สร้าง Frame สำหรับ Widgets ที่เกี่ยวข้องกับรูปภาพ
#frm_image_input = tk.Frame(root, padx=10, pady=10)
#frm_image_input.pack()

# Label สำหรับแสดงรูปภาพ
#lbl_img = tk.Label(frm_image_input)
#lbl_img.pack()

# Label และ Entry สำหรับเพศ
label_sex = tk.Label(root, text="SEX:", font=("Arial", 12), bg="#61f28d", fg="black")
label_sex.pack(pady=5)
entry_sex = tk.Entry(root)
entry_sex.pack(pady=5)

# Label และ Entry สำหรับอายุ
label_age = tk.Label(root, text="AGE:", font=("Arial", 12), bg="#61f28d", fg="black")
label_age.pack(pady=5)
entry_age = tk.Entry(root)
entry_age.pack(pady=5)

# Label และ Entry สำหรับน้ำหนัก
label_weight = tk.Label(root, text="WEIGHT:", font=("Arial", 12), bg="#61f28d", fg="black")
label_weight.pack(pady=5)
entry_weight = tk.Entry(root)
entry_weight.pack(pady=5)

# Label และ Entry สำหรับส่วนสูง
label_height = tk.Label(root, text="HEIGHT:", font=("Arial", 12), bg="#61f28d", fg="black")
label_height.pack(pady=5)
entry_height = tk.Entry(root)
entry_height.pack(pady=5)

# เพิ่ม Entry และ Label สำหรับการกรอกข้อมูลการสูบบุหรี่
label_smoking = tk.Label(root, text="SMOKING (0: No, 1: Yes):", font=("Arial", 12), bg="#61f28d", fg="black")
label_smoking.pack(pady=5)
entry_smoking = tk.Entry(root)
entry_smoking.pack(pady=5)

# เพิ่ม Entry และ Label สำหรับการกรอกข้อมูลการดื่มแอลกอฮอล์
label_drinking = tk.Label(root, text="DRINK ALCOHOL (0: No, 1: Yes):", font=("Arial", 12), bg="#61f28d", fg="black")
label_drinking.pack(pady=5)
entry_drinking = tk.Entry(root)
entry_drinking.pack(pady=5)

# สร้างปุ่มสำหรับเลือกไฟล์ CSV
btn_browse = tk.Button(frm_input, text="Browse CSV", command=browse_file, bg="#4CAF50", fg="white")
btn_browse.pack()

# สร้างปุ่มสำหรับเลือกรูปภาพ
#btn_browse_image = tk.Button(frm_image_input, text="Browse Image", command=browse_image, bg="#4CAF50", fg="black")
#btn_browse_image.pack()

# สร้างปุ่มสำหรับรีเซ็ตรูปภาพ
#btn_reset_image = tk.Button(frm_image_input, text="Reset Image", command=reset_image, bg="#FF0000", fg="white")
#btn_reset_image.pack()

# สร้างปุ่มสำหรับทำนาย BODY
btn_train_linear_regression = tk.Button(root, text="Predicted BODY", command=train_random_forest, bg="#4CAF50",
                                        fg="white")
btn_train_linear_regression.pack()

# สร้างปุ่มสำหรับรีเซ็ตข้อมูลที่ป้อน
btn_reset = tk.Button(root, text="Reset Inputs", command=reset_inputs, bg="#FF0000", fg="white")
btn_reset.pack()

# สร้าง Label สำหรับแสดงผลลัพธ์
result_var = tk.StringVar()
result_label = tk.Label(root, textvariable=result_var)
result_label.pack()


# ให้โปรแกรมแสดง GUI
root.mainloop()
