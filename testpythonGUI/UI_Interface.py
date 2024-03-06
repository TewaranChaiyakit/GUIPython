# Call lib
import test03
# Import necessary libraries
import tkinter as tk
from tkinter import ttk
from sklearn.metrics import accuracy_score

# Create a Tkinter window
window = tk.Tk()
window.title('Heart disease Dataset ')
window.configure(bg='#7efcf0')  # Set background color

# Get data from test03.py
data = test03.Voting.data

# Create a Frame for the left side
left_frame = tk.Frame(window, bg='#ffffff')  # Set background color for the right frame
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# Create a Treeview widget
tree = ttk.Treeview(left_frame, style="mystyle.Treeview")
tree["columns"] = tuple(data.columns)

# Add columns to Treeview
for col in data.columns:
    tree.column(col, anchor="center", width=100)
    tree.heading(col, text=col, anchor="center")

# Insert data into Treeview with alternating row colors
for index, row in data.iterrows():
    if index % 2 == 0:
        tree.insert("", index, values=tuple(row), tags=('even',))
    else:
        tree.insert("", index, values=tuple(row), tags=('odd',))

# Configure tag colors
tree.tag_configure('even', background='#b3e0ff')
tree.tag_configure('odd', background='#ffffff')

# Insert data into Treeview
for index, row in data.iterrows():
    tree.insert("", index, values=tuple(row))

# Style for Treeview
style = ttk.Style()
style.configure("mystyle.Treeview", highlightthickness=0, bd=0, font=('Arial', 10),
                rowheight=25)  # Set style for Treeview

# Pack the Treeview to the right frame
tree.pack(expand=tk.YES, fill=tk.BOTH)

# Create a Frame for the left side
right_frame = tk.Frame(window, bg='#345785')  # Set background color for the left frame
right_frame.pack(side=tk.RIGHT, padx=10, pady=20)

# Create labels and entry widgets for user input
input_labels = ['Chest_pain', 'Age', 'Weight', 'Height', 'Cholesterol', 'Max_hr']
entry_boxes = []

for label_text in input_labels:
    label = tk.Label(right_frame, text=label_text, font=("Arial", 12), bg='#fa7dfa')  # Set background color for labels
    label.grid(row=input_labels.index(label_text), column=0, pady=5, sticky='w')

    entry_var = tk.DoubleVar()  # Use DoubleVar for numeric input
    entry = tk.Entry(right_frame, textvariable=entry_var, font=("Arial", 12))
    entry.grid(row=input_labels.index(label_text), column=1, pady=5)

    entry_boxes.append(entry_var)

# Create a button to trigger predictions
predict_button = tk.Button(right_frame, text="Predict Heart_disease", command=lambda: predict_quality(entry_boxes),
                           bg='#4CAF50', fg='white')  # Set button color
predict_button.grid(row=len(input_labels), column=0, columnspan=2, pady=10)

# Create a label to display the predicted quality
result_label = tk.Label(right_frame, text="", font=("Arial", 14), bg='#b3e0ff')  # Set background color for the label
result_label.grid(row=len(input_labels) + 1, column=0, columnspan=2, pady=10)

# Create a label to display accuracy information
accuracy_labels = tk.Label(right_frame, text="", font=("Arial", 10), bg='#b3e0ff')  # Set background color for the label
accuracy_labels.grid(row=len(input_labels) + 2, column=0, columnspan=2, pady=10)


# Function to predict quality based on user input
def predict_quality(entry_boxes):
    # Get user input from entry boxes
    user_input = [entry.get() for entry in entry_boxes]

    # Make predictions using the Voting Classifier
    predictions = test03.Voting.voting_clf.predict([user_input])

    # Map numeric predictions to 'Good' or 'Bad'
    predicted_label = 'positive' if predictions[0] == 1 else 'negative'

    # Display the predicted quality
    result_label["text"] = f"Predicted Heart_disease: {predicted_label}"

    #

    # Display accuracy information for each classifier
    accuracy_labels["text"] = ""
    for model, model_instance in zip(
            ['Bagging', 'Boosting', 'Random Forest', 'Logistic', 'Naive Bayes', 'SVM', 'Decision Tree', 'MLP',
             'AdaBoost'],
            [test03.Voting.bagging_clf, test03.Voting.boosting_clf, test03.Voting.random_forest_model,
             test03.Voting.logistic_model, test03.Voting.naive_bayes_model, test03.Voting.svm_model,
             test03.Voting.decision_tree_model, test03.Voting.mlp_model, test03.Voting.adaboost_model]):
        predictions = model_instance.predict(test03.Voting.X)
        accuracy = accuracy_score(test03.Voting.y, predictions)
        if accuracy == 1:
            accuracy_labels["text"] += f"{model} Accuracy: Positive\n"
            accuracy_labels["fg"] = "green"
        else:
            accuracy_labels["text"] += f"{model} Accuracy: Negative\n"
            accuracy_labels["fg"] = "red"

# Run the Tkinter event loop
window.mainloop()



