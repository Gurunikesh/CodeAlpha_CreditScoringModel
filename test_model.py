import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")


def predict_credit():
    try:
        # Read numeric inputs
        age = float(entries["Age"].get())
        income = float(entries["Income"].get())
        debt = float(entries["Debt"].get())
        credit_score = float(entries["Credit Score"].get())
        loan_amount = float(entries["Loan Amount"].get())
        loan_term = float(entries["Loan Term"].get())
        num_cards = float(entries["Number of Credit Cards"].get())

        # Read dropdown inputs
        gender = entries["Gender"].get()
        education = entries["Education"].get()
        payment = entries["Payment History"].get()
        employment = entries["Employment Status"].get()
        residence = entries["Residence Type"].get()
        marital = entries["Marital Status"].get()

        # Encode categorical values
        gender = 0 if gender == "Male" else 1

        education = {
            "High School": 0,
            "Bachelor": 1,
            "Master": 2,
            "PhD": 3
        }[education]

        payment = {
            "Bad": 0,
            "Average": 1,
            "Good": 2
        }[payment]

        employment = {
            "Unemployed": 0,
            "Self-Employed": 1,
            "Employed": 2
        }[employment]

        residence = {
            "Rented": 0,
            "Owned": 1,
            "Mortgaged": 2
        }[residence]

        marital = {
            "Single": 0,
            "Married": 1,
            "Divorced": 2
        }[marital]

        # Prepare input array
        data = np.array([[
            age,
            gender,
            education,
            income,
            debt,
            credit_score,
            loan_amount,
            loan_term,
            num_cards,
            payment,
            employment,
            residence,
            marital
        ]])

        # Scale the input
        data = scaler.transform(data)

        # Predict
        result = model.predict(data)

        # Show result
        if result[0] == 1:
            result_label.config(text="Result: Creditworthy")
        else:
            result_label.config(text="Result: Not Creditworthy")

    except ValueError:
        messagebox.showerror(
            "Input Error",
            "Please enter valid numeric values."
        )
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"An error occurred:\n{e}"
        )


def clear_fields():
    # Clear numeric entry fields
    for field in fields:
        if field in dropdown_options:
            entries[field].set(dropdown_options[field][0])
        else:
            entries[field].delete(0, tk.END)

    result_label.config(text="Result: ")


# Create main window
root = tk.Tk()
root.title("Credit Scoring Prediction")
root.geometry("520x760")
root.configure(bg="white")
root.resizable(False, False)

# Title
title_label = tk.Label(
    root,
    text="Credit Scoring Prediction",
    font=("Arial", 18, "bold"),
    bg="white"
)
title_label.pack(pady=15)

# Main frame
frame = tk.Frame(root, bg="white")
frame.pack(pady=5)

# All input fields
fields = [
    "Age",
    "Gender",
    "Education",
    "Income",
    "Debt",
    "Credit Score",
    "Loan Amount",
    "Loan Term",
    "Number of Credit Cards",
    "Payment History",
    "Employment Status",
    "Residence Type",
    "Marital Status"
]

# Dropdown options
dropdown_options = {
    "Gender": ["Male", "Female"],
    "Education": ["High School", "Bachelor", "Master", "PhD"],
    "Payment History": ["Bad", "Average", "Good"],
    "Employment Status": ["Unemployed", "Self-Employed", "Employed"],
    "Residence Type": ["Rented", "Owned", "Mortgaged"],
    "Marital Status": ["Single", "Married", "Divorced"]
}

# Dictionary to store widgets/variables
entries = {}

# Create labels and input widgets
for i, field in enumerate(fields):
    label = tk.Label(
        frame,
        text=field,
        font=("Arial", 10),
        bg="white",
        anchor="w",
        width=25
    )
    label.grid(row=i, column=0, padx=10, pady=5, sticky="w")

    # Dropdown menu for categorical fields
    if field in dropdown_options:
        var = tk.StringVar()
        var.set(dropdown_options[field][0])

        option_menu = tk.OptionMenu(
            frame,
            var,
            *dropdown_options[field]
        )
        option_menu.config(width=27, font=("Arial", 10), bg="white")
        option_menu.grid(row=i, column=1, padx=10, pady=5)

        entries[field] = var

    # Entry box for numeric fields
    else:
        entry = tk.Entry(
            frame,
            width=30,
            font=("Arial", 10)
        )
        entry.grid(row=i, column=1, padx=10, pady=5)

        entries[field] = entry

# Button frame
button_frame = tk.Frame(root, bg="white")
button_frame.pack(pady=20)

# Predict button
predict_button = tk.Button(
    button_frame,
    text="Predict",
    font=("Arial", 12),
    width=15,
    command=predict_credit
)
predict_button.grid(row=0, column=0, padx=10)

# Clear button
clear_button = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 12),
    width=15,
    command=clear_fields
)
clear_button.grid(row=0, column=1, padx=10)

# Result label
result_label = tk.Label(
    root,
    text="Result: ",
    font=("Arial", 14, "bold"),
    bg="white"
)
result_label.pack(pady=10)

# Start GUI
root.mainloop()