import pandas as pd  # นำเข้าไลบรารี pandas เพื่อใช้ในการจัดการข้อมูล

from sklearn.ensemble import (  # นำเข้าคลาสตัวแบ่งชุดข้อมูลแบบองค์ประกอบ, แบบเพิ่มความแข็งแกร่ง, แบบป่าสุ่ม, และตัวจำแนกประเภทแบบโหวต
    BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
)

from sklearn.linear_model import LogisticRegression  # นำเข้าตัวจำแนกประเภทแบบโลจิสติกเรสเรียน
from sklearn.naive_bayes import GaussianNB  # นำเข้าตัวจำแนกประเภทแบบเบส์เธียนกาวาสเซียน
from sklearn.svm import SVC  # นำเข้าตัวจำแนกประเภทแบบเวกเตอร์สนับสนุน
from sklearn.tree import DecisionTreeClassifier  # นำเข้าตัวจำแนกประเภทแบบต้นไม้การตัดสินใจ
from sklearn.neural_network import MLPClassifier  # นำเข้าตัวจำแนกประเภทแบบพหุคุณชั้นเดียว
from sklearn.metrics import accuracy_score  # นำเข้าคำสั่ง accuracy_score เพื่อการประเมินประสิทธิภาพของโมเดล

class Voting :  # การกำหนดคลาส Voting
    # โหลดชุดข้อมูลจากไฟล์ CSV
    file_path = 'Heart_disease.csv'  # ที่อยู่ของไฟล์
    data = pd.read_csv(file_path)  # อ่านข้อมูลจากไฟล์ CSV โดยใช้ pandas

    # แยกคุณลักษณะและตัวแปรเป้าหมาย
    X = data[['Chest_pain', 'Age', 'Weight', 'Height', 'Cholesterol', 'Max_hr']]  # แยกคุณลักษณะ
    y = data['Disease'].astype('category').cat.codes  # แปลงป้ายกำกับเป็นค่าตัวเลข

    # การเตรียมโมเดลพื้นฐานและการฝึก
    logistic_model = LogisticRegression(max_iter=1000)  # การเตรียมโมเดลโลจิสติกเรสเรียน
    naive_bayes_model = GaussianNB()  # การเตรียมโมเดลเบส์เธียนกาวาสเซียน
    svm_model = SVC(probability=True)  # การเตรียมโมเดลเวกเตอร์สนับสนุนพร้อมการประเมินความน่าจะเป็น
    decision_tree_model = DecisionTreeClassifier()  # การเตรียมโมเดลต้นไม้การตัดสินใจ
    random_forest_model = RandomForestClassifier(n_estimators=50)  # การเตรียมโมเดลป่าสุ่ม
    mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000)  # การเตรียมโมเดลพหุคุณชั้นเดียว
    adaboost_model = AdaBoostClassifier()  # การเตรียมโมเดลเพิ่มความแข็งแกร่ง

    # การฝึกโมเดล
    logistic_model.fit(X, y)  # การฝึกโมเดลโลจิสติกเรสเรียน
    naive_bayes_model.fit(X, y)  # การฝึกโมเดลเบส์เธียนกาวาสเซียน
    svm_model.fit(X, y)  # การฝึกโมเดลเวกเตอร์สนับสนุน
    decision_tree_model.fit(X, y)  # การฝึกโมเดลต้นไม้การตัดสินใจ
    random_forest_model.fit(X, y)  # การฝึกโมเดลป่าสุ่ม
    mlp_model.fit(X, y)  # การฝึกโมเดลพหุคุณชั้นเดียว
    adaboost_model.fit(X, y)  # การฝึกโมเดลเพิ่มความแข็งแกร่ง

    # Bagging
    bagging_clf = BaggingClassifier(logistic_model, n_estimators=20)  # การสร้างโมเดลแบบ Bagging
    bagging_clf.fit(X, y)  # การฝึกโมเดลแบบ Bagging

    # Boosting
    boosting_clf = AdaBoostClassifier(logistic_model, n_estimators=30)  # การสร้างโมเดลแบบ Boosting
    boosting_clf.fit(X, y)  # การฝึกโมเดลแบบ Boosting

    # Voting Classifier
    voting_clf = VotingClassifier(  # การสร้างโมเดลตัวจำแนกประเภทแบบโหวต
        estimators=[
            ('bagging', bagging_clf),  # โมเดลแบบ Bagging
            ('boosting', boosting_clf),  # โมเดลแบบ Boosting
            ('random_forest', random_forest_model),  # โมเดลแบบป่าสุ่ม
            ('logistic', logistic_model),  # โมเดลแบบโลจิสติกเรสเรียน
            ('naive_bayes', naive_bayes_model),  # โมเดลแบบเบส์เธียนกาวาสเซียน
            ('svm', svm_model),  # โมเดลแบบเวกเตอร์สนับสนุน
            ('decision_tree', decision_tree_model),  # โมเดลแบบต้นไม้การตัดสินใจ
            ('mlp', mlp_model),  # โมเดลแบบพหุคุณชั้นเดียว
            ('adaboost', adaboost_model)  # โมเดลแบบเพิ่มความแข็งแกร่ง
        ], voting='hard'  # การตั้งค่าการโหวตเป็น 'hard' สำหรับการโหวตโดยความเห็นของส่วนใหญ่
    )
    voting_clf.fit(X, y)  # การฝึกโมเดลตัวจำแนกประเภทแบบโหวต
