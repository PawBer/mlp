import tkinter as tk
from PIL import Image
from typing import List
from network import *
import os
import random

class MLPCanvas(tk.Canvas):
    def __init__(self, master: tk.Widget, scale: int, network: Network, result_label: tk.Label, *args, **kwargs):
        self.scale: int = scale
        self.width: int = 28 * scale + 10
        self.height: int = 28 * scale + 10
        super().__init__(master, width=self.width, height=self.height, *args, **kwargs)

        self.drawn_pixels: List[float] = [0] * (28*28)
        self.network: Network = network
        self.result_label: tk.Label = result_label

        self.bind("<B1-Motion>", self.on_drag)

    def on_drag(self, event: tk.Event):
        x, y = event.x // self.scale, event.y // self.scale
        if x > 28 or y > 28 or x < 0 or y < 0:
            return
        
        #print(x, y)
        self.draw_pixel(x, y)
        self.drawn_pixels[(y - 1) * 28 + (x - 1)] = 1.0

        output = self.network.calculate_output(self.drawn_pixels)
        print(output)
        self.result_label.config(text=f"Widzę liczbę {output.index(max(output))}")

    def draw_pixel(self, x: int, y: int):
        x_scaled, y_scaled = x * self.scale, y * self.scale
        self.create_rectangle(x_scaled, y_scaled, x_scaled + self.scale, y_scaled + self.scale, fill="black")

    def clear_canvas(self):
        self.delete("all")
        self.drawn_pixels = [0] * (28*28)

if __name__ == "__main__":
    SCALE: int = 10
    root: tk.Tk = tk.Tk()

    with open("weights.json", "r") as weights:
        content = weights.read()
    network = Network.from_json(content)

    result_text: tk.Label = tk.Label(root, text="Narysuj liczbę")
    canvas: MLPCanvas = MLPCanvas(root, SCALE, network, result_text)
    canvas.pack()

    clear_button: tk.Button = tk.Button(root, text="Wyczyść", command=canvas.clear_canvas)
    clear_button.pack()

    result_text.pack()

    root.mainloop()