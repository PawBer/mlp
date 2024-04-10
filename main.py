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

    network = Network(28*28, 2, [4, 3])

    result_text: tk.Label = tk.Label(root, text="Narysuj liczbę")
    canvas: MLPCanvas = MLPCanvas(root, SCALE, network, result_text)
    canvas.pack()

    clear_button: tk.Button = tk.Button(root, text="Wyczyść", command=canvas.clear_canvas)
    clear_button.pack()

    result_text.pack()

    inputs: List[List[float]] = []
    targets: List[List[float]] = []

    for i in range(3):
        target_list: List[float] = []
        for j in range(3):
            target_list.append(0.0 if j != i else 1.0)

        for file_name in os.listdir(f"mnist/{i}/"):
            image: Image = Image.open(f"mnist/{i}/{file_name}")
            inputs.append(list(map(lambda pixel: 1.0 if (pixel / 255) > 0.5 else 0.0, image.getdata())))
            targets.append(target_list)

    shuffled_indices = list(range(len(inputs)))
    random.shuffle(shuffled_indices)

    inputs = [inputs[i] for i in shuffled_indices]
    targets = [targets[i] for i in shuffled_indices]

    test_inputs: List[List[float]] = []
    test_outputs: List[int] = []

    for i in range(3):
        for file_name in os.listdir(f"test/{i}/"):
            image: Image = Image.open(f"test/{i}/{file_name}")
            test_inputs.append(list(map(lambda pixel: 1.0 if (pixel / 255) > 0.5 else 0.0, image.getdata())))
            test_outputs.append(i)

    network.train(inputs_set=inputs, targets_set=targets, test_inputs=test_inputs, test_outputs=test_outputs, initial_learning_rate=0.05, epochs=100)

    #with open("weights.json", "w") as weights:
    #    weights.write(network.to_json())

    root.mainloop()