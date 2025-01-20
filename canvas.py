from tkinter import *
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageDraw
import PIL
import os
import image_preprocessing
import symbol_recognition
import equation_building_solving

# image_file='image_canvas/image.png'
# eqn = ""
# final_eqn = ""
# result = ""

# Constants
WIDTH = 1300
HEIGHT = 500
WHITE = (255, 255, 255)
LIGHT_GREEN = "#90EE90"  # Light green for active button
DEFAULT_BUTTON_BG = "#f0f0f0"  # Default button background
MIN_BRUSH_SIZE = 5
MAX_BRUSH_SIZE = 50

    # Initialize variables
brush_size_draw =  5 # Brush size for drawing (fixed size)
brush_size_eraser = 10  # Initial brush size for eraser
current_mode = "draw"  # Default mode is draw
cursor_id = None  # Canvas item ID for the cursor

def show_canvas(image_file) :
    # Initialize Tkinter
    root = Tk()
    root.title("SymSolver 🧾")

    # Canvas setup
    cv = Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
    cv.pack(expand=YES, fill=BOTH)

    # PIL Image setup
    image1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image1)

    #Function to calculate
    def calculate():
        save()
        image_preprocessing.image_preprocessing(image_file)
        global eqn, final_eqn
        eqn, final_eqn = symbol_recognition.symbol_recognition()
        extracted_eqn_entry.delete(0, END) 
        extracted_eqn_entry.insert(0, final_eqn)

    def solve():
        result = equation_building_solving.build_and_solve(eqn)
        dynamic_solution_label.config(text=result)





    # Function to save image
    def save() :
        try:
            directory = "image_canvas"  
            filename = os.path.join(directory, "image.png")
            
            image1.save(filename)
            # print(f"Image saved at: {filename} ")

        except Exception as e:
            # print(f"Error saving image: {e} ")
            messagebox.showerror("Error", f"Error saving image: {e}")

    # Paint function
    def paint(event):
        global current_mode
        color = "white" if current_mode == "eraser" else "black"  # Use white for eraser mode
        brush_size = brush_size_eraser if current_mode == "eraser" else brush_size_draw  # Use eraser size or draw size
        x1, y1 = (event.x - brush_size), (event.y - brush_size)
        x2, y2 = (event.x + brush_size), (event.y + brush_size)
        
        # Draw on canvas
        cv.create_oval([x1, y1, x2, y2], fill=color, outline=color, width=0)
        
        # Draw on the image
        draw.ellipse([x1, y1, x2, y2], fill=color, outline=color)

    # Set mode (draw or eraser)
    def set_mode(mode):
        global current_mode
        current_mode = mode
        if mode == "draw":
            draw_button.config(bg=LIGHT_GREEN)
            eraser_button.config(bg=DEFAULT_BUTTON_BG)
            # print("Switched to Draw mode ")
        elif mode == "eraser":
            eraser_button.config(bg=LIGHT_GREEN)
            draw_button.config(bg=DEFAULT_BUTTON_BG)
            # print("Switched to Eraser mode ")

    # Update brush size
    def update_brush_size(increment):
        global brush_size_eraser
        brush_size_eraser = max(MIN_BRUSH_SIZE, min(MAX_BRUSH_SIZE, brush_size_eraser + increment))
        # print(f"Eraser size updated to: {brush_size_eraser}px ")

    # Clear canvas
    def clear_canvas():
        cv.delete("all")
        draw.rectangle([0, 0, WIDTH, HEIGHT], fill=WHITE)
        extracted_eqn_entry.delete(0, END) 
        dynamic_solution_label.config(text="")


    # Update cursor
    def update_cursor(event):
        global cursor_id
        brush_size = brush_size_eraser if current_mode == "eraser" else brush_size_draw  # Use eraser size or draw size
        x, y = event.x, event.y
        x1, y1 = x - brush_size, y - brush_size
        x2, y2 = x + brush_size, y + brush_size

        if cursor_id is not None:
            cv.delete(cursor_id)  # Remove old cursor

        cursor_color = "gray" if current_mode == "eraser" else "black"
        cursor_id = cv.create_oval(x1, y1, x2, y2, outline=cursor_color, width=2)

    # UI setup
    # Canvas for drawing
    cv.pack(expand=YES, fill=BOTH)

    intro_frame = Frame(root)
    intro_frame.pack(fill=X, pady=3)

    intro_label = Label(
        intro_frame, 
        text="Supported symbols: 0 - 9, (, ), a, b, c, y, =, π",
        font=('Helvetica', 10), 
        wraplength=600,  # Wrapping text for better readability
        justify=LEFT  # Center-aligning the text
    )
    intro_label.pack(pady=3)

    # Frame for extracted equation and solution
    equation_solution_frame = Frame(root)
    equation_solution_frame.pack(fill=X, pady=10)

    # Extracted Equation Section
    extracted_eqn_label = Label(equation_solution_frame, text="Extracted Equation: ", font=('Helvetica', 12))
    extracted_eqn_label.pack(side=LEFT, padx=5, pady=5)  # Added some vertical padding
    extracted_eqn_entry = Entry(equation_solution_frame, font=('Helvetica', 12), width=50)
    extracted_eqn_entry.pack(side=LEFT, padx=10, pady=5)  # Added some vertical padding


    # Solution Section
    solution_label = Label(equation_solution_frame, text="Solution: ", font=('Helvetica', 12))
    solution_label.pack(side=LEFT, padx=5, pady=5)  # Added some vertical padding

    # New dynamic content label beside solution label
    dynamic_solution_label = Label(equation_solution_frame, text="", font=('Helvetica', 12))
    dynamic_solution_label.pack(side=LEFT, padx=10, pady=5)

    # Button panel for actions
    button_frame = Frame(root)
    button_frame.pack(side=BOTTOM, fill=X)

    # Centered buttons container
    button_container = Frame(button_frame)
    button_container.pack(side=TOP, fill=X)

    # Left-aligned buttons (Draw, Eraser, +, -)
    left_buttons_frame = Frame(button_container)
    left_buttons_frame.pack(side=LEFT, padx=20, pady=10)

    draw_button = Button(left_buttons_frame, text="Draw", command=lambda: set_mode("draw"), bg=LIGHT_GREEN,font=('Helvetica', 12))
    draw_button.pack(side=LEFT, padx=5, pady=5)

    eraser_button = Button(left_buttons_frame, text="Eraser", command=lambda: set_mode("eraser"),font=('Helvetica', 12))
    eraser_button.pack(side=LEFT, padx=5, pady=5)

    eraser_plus_button = Button(left_buttons_frame, text="Eraser +", command=lambda: update_brush_size(2),font=('Helvetica', 12))
    eraser_plus_button.pack(side=LEFT, padx=5, pady=5)

    eraser_minus_button = Button(left_buttons_frame, text="Eraser -", command=lambda: update_brush_size(-2),font=('Helvetica', 12))
    eraser_minus_button.pack(side=LEFT, padx=5, pady=5)

    # Calculate button in the center
    extract_button = Button(button_container, text="Calculate", command=calculate ,font=('Helvetica', 12))
    extract_button.pack(side=LEFT, padx=200, pady=5)
    calculate_button = Button(button_container, text="Solve", command=solve ,font=('Helvetica', 12))
    calculate_button.pack(side=LEFT, pady=5)

    # Clear button on the right
    clear_button = Button(button_container, text="Clear", command=clear_canvas,font=('Helvetica', 12))
    clear_button.pack(side=RIGHT, padx=20, pady=5)

    # Bind mouse events to paint and cursor update
    cv.bind("<B1-Motion>", paint)
    cv.bind("<Button-1>", paint)
    cv.bind("<Motion>", update_cursor)

    # Start the application
    root.mainloop()
