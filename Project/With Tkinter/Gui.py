import sys
sys.path.append('C:\\Users\\pc\\Desktop\\OCR')
import tkinter as tk
import sympy
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from sympy import symbols, Eq, solve
from sympy.parsing.sympy_parser import parse_expr
from ttkthemes import ThemedTk
import re
import os
import shutil
from keras.models import load_model
import pandas as pd
import numpy as np
from PIL import Image,ImageOps
import CharacterSegmentation as cs
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cv2
import numpy as np

original_image = None
photo = None
image_canvas = None
image_path = None

def clear_folder(folder_path):
    for item_name in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item_name)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f'Failed to delete {item_path}. Reason: {e}')

def insert_multiplication_symbol(equation_str):
    pattern_num_letter = r"(\d)([a-zA-Z])"
    modified_equation = re.sub(pattern_num_letter, r"\1*\2", equation_str)
    
    pattern_letter_num = r"([a-zA-Z])(\d)"
    modified_equation = re.sub(pattern_letter_num, r"\1**\2", modified_equation)
    
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
        
        solutions = solve(equation, (x, y, z))
    else:
        solutions = parse_expr(equation_str)
    
    if(type(solutions) == list):
        result = "The solutions are: "
        if(str(solutions[0][0]) != 'x'):
            result += "x = " + str(solutions[0][0])
        if(str(solutions[0][1]) != 'y'):
            result += "y = " + str(solutions[0][1])
        if(str(solutions[0][2]) != 'z'):
            result += "z = " + str(solutions[0][2])
        return result
    elif(type(solutions) == sympy.core.numbers.Integer):
        result = "The solution is: " + str(solutions)
        return result
    else:
        return "None"
    
def img2emnist(filepath, char_code):
    img = Image.open(filepath).resize((28, 28))
    inv_img = ImageOps.invert(img)
    
    flatten = np.array(inv_img).flatten()
    flatten = flatten / 255
    flatten = np.where(flatten > 0.5, 1, 0)
    
    csv_img = ','.join([str(num) for num in flatten])
    csv_str = '{},{}'.format(char_code, csv_img)
    return csv_str
    
def run_model(image_path):
    INPUT_IMAGE = image_path
    SEGMENTED_OUTPUT_DIR = 'C:\\Users\\pc\\Desktop\\OCR\\segmented/'
    MODEL_PATH = 'C:\\Users\\pc\\Desktop\\OCR\\model\\modelOCR3.h5'
    mapping_processed = 'C:\\Users\\pc\\Desktop\\OCR\\data\\eminst\\processed-mapping_filtered_renumbered.csv'
    cs.image_segmentation(INPUT_IMAGE)
    segmented_images = []
    files = [f for r, d, f in os.walk(SEGMENTED_OUTPUT_DIR)][0]
    for f in files:
        segmented_images.append(Image.open(SEGMENTED_OUTPUT_DIR + f))
    size = len(segmented_images)
    for i in range(size):
        img = segmented_images[i]
    files = [f for r, d, f in os.walk(SEGMENTED_OUTPUT_DIR)][0]
    for f in files:
        filename = SEGMENTED_OUTPUT_DIR + f
        img = cv2.imread(filename, 0)
        kernel = np.ones((2,2), np.uint8)
        dilation = cv2.erode(img, kernel, iterations = 1)
        cv2.imwrite(filename, dilation)
    temp_filename = 'test.csv'
    if os.path.exists(temp_filename):
        os.remove(temp_filename)
    f_test = open(temp_filename, 'a+')
    column_names = ','.join(["label"] + ["pixel" + str(i) for i in range(784)])
    print(column_names, file=f_test)
    files = [f for r, d, f in os.walk(SEGMENTED_OUTPUT_DIR)][0]
    for f in files:
        csv = img2emnist(SEGMENTED_OUTPUT_DIR + f, -1)
        print(csv, file=f_test)
    f_test.close()
    test_df = pd.read_csv(temp_filename)
    data = pd.read_csv(temp_filename)
    X_data = data.drop(labels = ["label"], axis = 1)
    X_data = X_data.values.reshape(-1,28,28,1)
    size = X_data.shape[0]
    for i in range(size):
        v = X_data[i][:,:,0].astype('uint8')
        img = Image.fromarray(255* v)
    df = pd.read_csv(mapping_processed)
    code2char = {}
    for index, row in df.iterrows():
        code2char[row['id']] = row['char']
    model = load_model(MODEL_PATH)
    results = model.predict(X_data)
    results = np.argmax(results, axis = 1)
    parsed_str = ""
    for r in results:
        parsed_str += code2char[r]

    return parsed_str

def process_image():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Ask the user to select an image file
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif")])

    if image_path:
        print("Selected image path:", image_path)
        clear_folder('C:\\Users\\pc\\Desktop\\OCR\\segmented')
        modeledEquation = run_model(image_path)
        modeled_label.config(text=str(modeledEquation))
        solution = evaluate_equation(modeledEquation)
        solution_label.config(text=str(solution))
    else:
        print("No image selected.")


def solve_equation():
    equation = equation_entry.get()
    try:
        solution = evaluate_equation(equation)
        solution_label.config(text=str(solution))
    except Exception as e:
        solution_label.config(text=str(e))

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
solve_button = ttk.Button(mainframe, text="Solve Equation", command=solve_equation)
solve_button.grid(column=3, row=3, sticky=(tk.W))

# Create "Process Image" button for image equation
process_image_button = ttk.Button(mainframe, text="Process Image", command=process_image)
process_image_button.grid(column=2, row=2, sticky=(tk.W))

# Create label to display modeled equation
modeled_label = ttk.Label(mainframe, text="")
modeled_label.grid(column=1, row=5, columnspan=4, sticky=(tk.W))

# Create label to display solution
solution_label = ttk.Label(mainframe, text="")
solution_label.grid(column=1, row=10, columnspan=4, sticky=(tk.W))


root.mainloop() 