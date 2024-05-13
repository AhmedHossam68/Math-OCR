import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from sympy import symbols, Eq, solve
from sympy.parsing.sympy_parser import parse_expr
from ttkthemes import ThemedTk
from keras.models import load_model
import numpy as np
import re

# Load the model
model = load_model('modelOCR3.h5')

original_image = None
photo = None
image_canvas = None

def insert_multiplication_symbol(equation_str):
    pattern = r"(\d)([a-zA-Z])"
    modified_equation = re.sub(pattern, r"\1*\2", equation_str)
    return modified_equation

def evaluate_equation(equation_str):
    equation_str = insert_multiplication_symbol(equation_str)
    
    if '=' in equation_str:
        x, y, z = symbols('x y z')
        
        equation_str = equation_str.replace('X', 'x').replace('Y', 'y').replace('Z', 'z')
        
        lhs, rhs = equation_str.split('=')
        
        lhs_expr = parse_expr(lhs)

        rhs_expr = parse_expr(rhs)
        
        equation = Eq(lhs_expr, rhs_expr)
        
        # Solve the equation
        solution = solve(equation)
        
        return solution

def process_image():
    global original_image, photo, image_canvas
    filename = filedialog.askopenfilename()
    if filename:
        original_image = Image.open(filename)
        photo = ImageTk.PhotoImage(original_image)
        if image_canvas is None:
            image_canvas = tk.Canvas(root, width=photo.width(), height=photo.height())
            image_canvas.pack(pady=20)
        else:
            image_canvas.config(width=photo.width(), height=photo.height())
        image_canvas.create_image(photo.width()//2, photo.height()//2, image=photo)

    # Preprocess the image for the model
    # This step depends on how your model expects the input. 
    # Here's an example of converting the image to grayscale and resizing it to 28x28.
    processed_image = original_image.convert('L').resize((28, 28))

    # Convert the image to a numpy array and reshape it to match the model's input shape
    input_data = np.array(processed_image).reshape(1, 28, 28, 1)

    # Use the model to make a prediction
    prediction = model.predict(input_data)

    # Update the solution label with the prediction
    solution_label.config(text=str(prediction))

root = ThemedTk(theme="arc")
root.title("Math Equation Solver")
root.geometry('800x600')

mainframe = ttk.Frame(root, padding="10")
mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

# Create entry field for equation
equation_label = ttk.Label(mainframe, text="Enter your equation:")
equation_label.grid(column=1, row=1, sticky=(tk.W))
equation_entry = ttk.Entry(mainframe, width=30)
equation_entry.grid(column=2, row=1, sticky=(tk.W, tk.E))

# Create "Solve" button for manual equation
solve_button = ttk.Button(mainframe, text="Solve Equation", command=evaluate_equation)
solve_button.grid(column=3, row=1, sticky=(tk.W))

# Create "Process Image" button for image equation
process_image_button = ttk.Button(mainframe, text="Process Image", command=process_image)
process_image_button.grid(column=2, row=2, sticky=(tk.W))

# Create label to display solution
solution_label = ttk.Label(mainframe, text="")
solution_label.grid(column=1, row=3, columnspan=4, sticky=(tk.W))

root.mainloop()