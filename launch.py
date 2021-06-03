import tkinter as tk
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from load_state import prepare_image
from utils import draw_state
from blocks_world import BlocksWorld
from search_algs import *

# file che contiene l'implementazione dell'interfaccia grafica per utilizzare il programma

class Window(tk.Frame):


    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.initial_state = None
        self.goal_state = None
        self.create_widgets()
        self.create_images("insert_image.png", "insert_image.png")


    def create_widgets(self):
        initial_label = tk.Label(self, text = "Seleziona stato iniziale:")
        goal_label = tk.Label(self, text = "Seleziona stato finale:")
        initial_label.grid(row = 0, column = 0, padx = 10, pady = 10)
        goal_label.grid(row = 0, column = 2, padx = 10, pady = 10)

        initial_button = tk.Button(self, text="Seleziona file", command=self.open_initial)
        goal_button = tk.Button(self, text="Seleziona file", command=self.open_goal)
        initial_button.grid(row = 1, column = 0, padx = 10, pady = 10)
        goal_button.grid(row = 1, column = 2, padx = 10, pady = 10)

        alg_label = tk.Label(self, text = "Seleziona algoritmo di ricerca:")
        alg_label.grid(row = 0, column = 1, padx = 10, pady = 10)

        frame = tk.Frame(self)
        frame.grid(row = 1, column = 1, padx = 10, pady = 10)

        self.selected = tk.StringVar(self)
        self.selected.set("BFS")
        select_alg_menu = tk.OptionMenu(frame, self.selected, "BFS", "DFS", "IDS", "UCS", "A*", "RBFS", command=self.read_algorithm).pack()

        start_button = tk.Button(frame, text="Start search", command=self.start_search).pack()


    def create_images(self, initial, goal):
        self.initial_image_path = initial
        self.initial_image = ImageTk.PhotoImage(Image.open("./images/" + initial).resize((300, 300)))
        initial_image_label = tk.Label(self, image=self.initial_image)
        initial_image_label.grid(row = 2, column = 0, padx = 10, pady = 10)

        self.goal_image_path = goal
        self.goal_image = ImageTk.PhotoImage(Image.open("./images/" + goal).resize((300, 300)))
        goal_image_label = tk.Label(self, image=self.goal_image)
        goal_image_label.grid(row = 2, column = 2, padx = 10, pady = 10)


    def open_initial(self):
        self.initial_file = askopenfilename()
        if self.initial_file == ():
            return
        self.initial_state = prepare_image(self.initial_file, False)
        print(self.initial_state)
        draw_state(self.initial_state, "initial")
        self.create_images("/temp/initial.jpg", self.goal_image_path)


    def read_algorithm(self, alg):
        return alg


    def open_goal(self):
        self.goal_file = askopenfilename()
        if self.goal_file == ():
            return
        self.goal_state = prepare_image(self.goal_file, False)
        print(self.goal_state)
        draw_state(self.goal_state, "goal")
        self.create_images(self.initial_image_path, "/temp/goal.jpg")


    def start_search(self):
        if self.goal_state is None and self.initial_state is None:
            return
        alg = self.selected.get()
        problem = BlocksWorld(self.initial_state, self.goal_state)
        print("Inizio ricerca:")
        if alg == "BFS":
            problem.solution(graph_bfs(problem).solution())
        if alg  == "A*":
            problem.solution(a_star(problem, lambda n: problem.misplaced_blocks(n)).solution())
        if alg == "DFS":
            problem.solution(graph_dfs(problem).solution())
        if alg  == "IDS":
            problem.solution(ids(problem).solution())
        if alg == "RBFS":
            problem.solution(rbfs(problem, lambda n: problem.misplaced_blocks(n)).solution())
        if alg == "UCS":
            problem.solution(a_star(problem, lambda n: problem.depth(n)).solution())

            


root = tk.Tk()
root.title("Blocks World")
root.resizable(0, 0) 
app = Window(master=root)
app.mainloop()