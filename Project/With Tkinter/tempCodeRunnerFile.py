import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from sympy import symbols, Eq, solve
from sympy.parsing.sympy_parser import parse_expr
import re
from tkinter import PhotoImage, Label

# Rest of your code

original_image = None
photo = None
image_canvas = None
button_image = None

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
    # Call your AI model here with original_image as input
    # For example:
    # solution = ai_model.solve(original_image)
    solution_label.config(text=str(solution))

def solve_equation():
    equation = equation_entry.get()
    try:
        solution = evaluate_equation(equation)
        solution_label.config(text=str(solution))
    except Exception as e:
        solution_label.config(text=str(e))

root = tk.Tk()
root.title("Niggasaurus Rex OCR")
root.geometry('800x600')

# Load the original image file
original_image = Image.open("C:\\Uni\\Junior Sem2\\AI\\Project\\Math-OCR\\Project\\With Tkinter\\Screenshot 2024-05-14 003729.png")

# Create a canvas for the background image
bg_canvas = tk.Canvas(root, width=800, height=600)
bg_canvas.pack(fill="both", expand=True)

# Add the image to the canvas
bg_image = ImageTk.PhotoImage(original_image)
bg_canvas.create_image(0, 0, image=bg_image, anchor="nw")

# Create a smaller version of the image for the buttons
button_image = bg_image.subsample(10, 10)  # adjust the numbers to change the size of the image

def resize_image(event):
    # Resize the image to the size of the application window
    new_width = event.width
    new_height = event.height
    new_image = original_image.resize((new_width, new_height))
    
    # Convert the image to a PhotoImage
    bg_image = ImageTk.PhotoImage(new_image)

    # Update the image on the canvas
    bg_canvas.create_image(0, 0, image=bg_image, anchor="nw")
    bg_canvas.image = bg_image  # keep a reference to the image

# Bind the resize_image function to the <Configure> event
root.bind('<Configure>', resize_image)

mainframe = ttk.Frame(bg_canvas, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))

# Create entry field for equation
equation_label = ttk.Label(mainframe, text="Enter your equation:")
equation_label.grid(column=1, row=1, sticky=(tk.W))
equation_entry = ttk.Entry(mainframe, width=30)
equation_entry.grid(column=2, row=1, sticky=(tk.W, tk.E))

# Create "Solve" button for manual equation
solve_button = ttk.Button(mainframe, text="Solve Equation", command=solve_equation, image=button_image, compound="center")
solve_button.grid(column=3, row=1, sticky=(tk.W))

# Create "Process Image" button for image equation
process_image_button = ttk.Button(mainframe, text="Process Image", command=process_image, image=button_image, compound="center")
process_image_button.grid(column=2, row=2, sticky=(tk.W))

# Create label to display solution
solution_label = ttk.Label(mainframe, text="")
solution_label.grid(column=1, row=2, columnspan=4, sticky=(tk.W))

from PIL import Image

# Load the original image file
original_image = Image.open("C:\\Uni\\Junior Sem2\\AI\\Project\\Math-OCR\\Project\\With Tkinter\\Screenshot 2024-05-14 003729.png")

# Resize the image
smaller_image = original_image.resize((50, 50))  # adjust the numbers to change the size of the image

# Convert the smaller image to a PhotoImage
button_image = ImageTk.PhotoImage(smaller_image)

# Now you can use button_image for your buttons

root.mainloop()