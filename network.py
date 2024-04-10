from typing import List, Optional
import random
import json

def relu(x):
    if x >= 0:
        return x
    else:
        return x * 0.1

def relu_derivative(x):
    if x >= 0:
        return 1
    else:
        return 0.1

class Neuron:
    def __init__(self, input_amount: int, weights: Optional[List[float]] = None, bias: Optional[float] = None):
        stddev: float = (2/input_amount) ** 0.5
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [random.gauss(0, stddev) for _ in range(input_amount)]

        if bias is not None:
            self.bias = bias
        else:
            self.bias: float = random.gauss(0, stddev)


    def calculate_output(self, inputs: List[float]) -> float:
        sum: float = 0.0
        sum += self.bias

        for weight, input in zip(self.weights, inputs):
            sum += weight * input

        return relu(sum)


class Layer:
    def __init__(self, neuron_amount: int, input_amount: int, neurons: Optional[List[Neuron]] = None):
        self.input_amount: int = input_amount
        if neurons is not None:
            self.neurons = neurons
        else:
            self.neurons = [Neuron(input_amount) for _ in range(neuron_amount)]

    def calculate_output(self, inputs: List[float]) -> List[float]:
        outputs: List[float] = []

        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))

        return outputs


class Network:
    def __init__(self, input_amount: int, layer_amount: int, neurons_in_layer: List[int], layers: Optional[List[Layer]] = None):
        if layers is not None:
            self.layers = layers
        else:
            self.layers: List[Layer] = []
            for i in range(layer_amount):
                self.layers.append(
                    Layer(
                        neurons_in_layer[i],
                        input_amount if i == 0 else neurons_in_layer[i - 1],
                    )
                )
        
    
    def calculate_output(self, inputs: List[float]) -> List[float]:
        output: List[float] = []

        for layer in self.layers:
            inputs = output = layer.calculate_output(inputs)

        return output
    
    def forward_propagation(self, inputs: List[float]) -> List[List[float]]:
        outputs: List[List[float]] = []

        for layer in self.layers:
            inputs = layer.calculate_output(inputs)
            outputs.append(inputs)

        return outputs
    
    def calculate_mse(self, inputs: List[List[float]], targets: List[List[float]]) -> float:
        total_mse: float = 0.0
        for inputs, targets in zip(inputs, targets):
            outputs: float = self.calculate_output(inputs)
            total_mse += sum((targets[i] - outputs[i]) ** 2 for i in range(len(targets))) / 2.0
        return total_mse / len(inputs)
    
    def calculate_accuracy(self, test_inputs: List[List[float]], test_outputs: List[List[float]]) -> float:
        correct_predictions: int = 0
        total_predictions: int = len(test_inputs)

        for i in range(len(test_inputs)):
            output: List[float] = self.calculate_output(test_inputs[i])
            predicted_digit: int = output.index(max(output))
            actual_digit: int = test_outputs[i]
            if predicted_digit == actual_digit:
                correct_predictions += 1

        return (correct_predictions / total_predictions) * 100

    def backward_propagation(self, inputs: List[float], targets: List[float], learning_rate: float):
        # Forward propagation
        outputs: List[List[float]] = self.forward_propagation(inputs)
        
        # Compute errors for output layer
        errors: List[float] = [targets[i] - outputs[-1][i] for i in range(len(targets))]
        
        # Create deltas
        deltas: List[List[float]] = [[] for _ in range(len(self.layers))]
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            if i == len(self.layers) - 1:  # Output layer
                for j in range(len(layer.neurons)):
                    deltas[i].append(errors[j] * relu_derivative(outputs[i][j]))
            else:  # Other layers
                for j, neuron in enumerate(layer.neurons):
                    delta: float = 0.0
                    for next_layer_neuron_idx, next_layer_neuron in enumerate(self.layers[i+1].neurons):
                        delta += next_layer_neuron.weights[j] * deltas[i+1][next_layer_neuron_idx]
                    deltas[i].append(delta * relu_derivative(outputs[i][j]))

        # Modify weights
        for layer_idx, layer in enumerate(self.layers):
            for neuron_idx, neuron in enumerate(layer.neurons):
                for weight_idx, _ in enumerate(neuron.weights):
                    if layer_idx == 0:
                        neuron.weights[weight_idx] += learning_rate * deltas[layer_idx][neuron_idx] * inputs[weight_idx]
                    else:
                        neuron.weights[weight_idx] += learning_rate * deltas[layer_idx][neuron_idx] * outputs[layer_idx - 1][weight_idx]
                neuron.bias += learning_rate * deltas[layer_idx][neuron_idx]

    def train(self, inputs_set: List[List[float]], targets_set: List[List[float]], test_inputs: List[List[float]], test_outputs: List[List[float]], initial_learning_rate: float, epochs: int):
        mse: float = self.calculate_mse(inputs_set, targets_set)
        accuracy: float = self.calculate_accuracy(test_inputs, test_outputs)
        print(f"MSE before training: {mse}, Accuracy before training: {accuracy}")

        for epoch in range(epochs):
            learning_rate: float = initial_learning_rate * (0.98 ** epoch)

            for inputs, targets in zip(inputs_set, targets_set):
                self.backward_propagation(inputs, targets, learning_rate)

            accuracy: float = self.calculate_accuracy(test_inputs, test_outputs)
            mse: float = self.calculate_mse(inputs_set, targets_set)
            print(f"Epoch {epoch+1}/{epochs}, MSE: {mse}, Accuracy: {accuracy} Learning Rate: {learning_rate}")

    @classmethod
    def from_json(cls, json_string: str):
        network_data = json.loads(json_string)

        layers: List[Layer] = []
        for layer_json in network_data["layers"]:
            neurons: List[Neuron] = []
            for neuron_json in layer_json["neurons"]:
                neuron = Neuron(len(neuron_json["weights"]), neuron_json["weights"], neuron_json["bias"])
                neurons.append(neuron)
            layer = Layer(len(neurons), len(neurons[0].weights), neurons)
            layers.append(layer)

        return cls(len(layers[0].neurons[0].weights), len(layers), [len(layer.neurons) for layer in layers], layers=layers)
    
    def to_json(self) -> str:
        network_json = {
            "layers": []
        }

        for layer in self.layers:
            layer_json = {
                "neurons": []
            }

            for neuron in layer.neurons:
                neuron_json = {
                    "weights": neuron.weights,
                    "bias": neuron.bias
                }
                layer_json["neurons"].append(neuron_json)

            network_json["layers"].append(layer_json)

        return json.dumps(network_json, indent=4)