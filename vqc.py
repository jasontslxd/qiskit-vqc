import time
from multiprocessing import Pool
from functools import partial
import numpy as np
from qiskit import Aer, transpile, QuantumCircuit
from qiskit.providers import Backend
from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.circuit.library.n_local import EfficientSU2
from qiskit_machine_learning.circuit.library import RawFeatureVector
from sklearn.metrics import log_loss, accuracy_score
from optimiser import Optimiser

class VQC():
    def __init__(self, optimiser:Optimiser, n_features:int=256, n_data:int=8, n_output:int=1, n_layers:int=2, 
                 backend:Backend=None) -> None:
        self.optimiser = optimiser
        self.n_features = n_features
        self.n_data = n_data
        self.n_output = n_output
        self.n_qubits = n_data + n_output
        self.n_layers = n_layers
        
        if backend is None:
            self.backend = Aer.get_backend("aer_simulator_statevector")
            # try:
            #     self.backend = Aer.get_backend("aer_simulator_statevector_gpu")
            #     print("GPU simulator available, using GPU")
            # except QiskitBackendNotFoundError:
            #     self.backend = Aer.get_backend("aer_simulator_statevector")
            #     print("GPU simulator unavailable, using CPU")
        else:
            self.backend = backend

        parameterized_ansatz = EfficientSU2(self.n_qubits, su2_gates=['rz', 'rx', 'rz'], 
                                            entanglement='linear', reps=n_layers, 
                                            insert_barriers=True, skip_final_rotation_layer=True)
        self.transpiled_ansatz = transpile(parameterized_ansatz, self.backend)
        self.n_parameters = self.transpiled_ansatz.num_parameters
        print("Number of trainable parameters: {}".format(self.n_parameters))
        self.parameters = np.random.uniform(-np.pi, np.pi, (self.n_parameters,))

    def generate_transpiled_circuit(self, x:np.ndarray, parameters:np.ndarray):
        circuit = QuantumCircuit(self.n_qubits, self.n_output)
        amplitude_encoding = RawFeatureVector(self.n_features).assign_parameters(x)
        ansatz = self.transpiled_ansatz.assign_parameters(parameters)

        circuit.compose(amplitude_encoding, inplace=True)
        circuit.x(range(self.n_data, self.n_qubits))
        circuit.compose(ansatz, inplace=True)
        circuit.save_statevector()

        # circuit.draw(output='mpl', filename="circuit")

        transpiled = transpile(circuit, self.backend)

        return transpiled

    def output_probability(self, circuit:QuantumCircuit) -> float:
        # samples probability of measuring label 1
        result = self.backend.run(circuit, shots=1).result()
        output_state = result.get_statevector(circuit).probabilities([x for x in range(self.n_data, self.n_qubits)])
        prob1 = output_state[1]
        return prob1

    def predict(self, xs:np.ndarray, parameters:np.ndarray=None):
        # generates predictions for array of inputs
        if parameters is None:
            parameters = self.parameters
        if xs.shape[0] == 1:
            predictions = [self.predict_sample(xs.reshape(256,), parameters)] 
        else:
            predictions = [self.predict_sample(sample, parameters) for sample in xs]
        return np.array(predictions, dtype=np.float64)

    def predict_parallel(self, xs:np.ndarray, pool:Pool, parameters:np.ndarray=None):
        if parameters is None:
            parameters = self.parameters
        if xs.shape[0] == 1:
            return np.array([self.predict_sample(xs, parameters)], dtype=np.float64)
        else:
            # Need add map with parameters
            predictions = pool.map(self.predict_sample, xs)
            return np.array(predictions, dtype=np.float64)

    def predict_sample(self, x:np.ndarray, parameters:np.ndarray=None):
        if parameters is None:
            parameters = self.parameters
        circuit = self.generate_transpiled_circuit(x, parameters)
        prob1 = self.output_probability(circuit)
        return prob1

    def compute_loss(self, xs:np.ndarray, ys:np.ndarray, parameters: np.ndarray) -> float:
        predictions = self.predict(xs, parameters)
        return log_loss(ys, predictions, labels=[0,1])

    def accuracy(self, labels:np.ndarray, predictions:np.ndarray) -> float:
        # predictions should be array of probabilities of predicting label 1
        assert labels.shape[0] == predictions.shape[0]
        count = 0
        for label, prediction in zip(labels, predictions):
            if (label == 1 and prediction >= 0.5) or (label == 0 and prediction < 0.5):
                count += 1
        return count / labels.shape[0]

    def gradient(self, xs:np.ndarray, ys:np.ndarray, pool:Pool=None) -> np.ndarray:
        # returns the gradient vector
        if not pool is None:
            func = partial(self.parameter_gradient, xs, ys)
            gradient_vector = pool.map(func, range(self.n_parameters))
        else:
            gradient_vector = []
            for i in range(self.n_parameters):
                gradient_vector.append(self.parameter_gradient(xs, ys, i))
            gradient_vector = np.array(gradient_vector, dtype=np.float64)
        return np.array(gradient_vector, dtype=np.float64)

    def parameter_gradient(self, xs:np.ndarray, ys:np.ndarray, i:int) -> float:
        # computes gradient of single parameter
        shifted_params = np.copy(self.parameters)
        shifted_params[i] += np.pi / 2
        upper_loss = self.compute_loss(xs, ys, shifted_params)
        shifted_params[i] -= np.pi
        lower_loss = self.compute_loss(xs, ys, shifted_params)
        return 0.5 * (upper_loss - lower_loss)

    def fit(self, xs:np.ndarray, ys:np.ndarray, epochs:int=100) -> None:
        pool = Pool()
        for i in range(epochs):
            start = time.time()
            gradient_vector = self.gradient(xs, ys, pool)
            new_params = self.optimiser.step(self.parameters, gradient_vector)
            diff = np.average(np.absolute(np.subtract(new_params, self.parameters)))
            self.parameters = new_params
            new_predictions = self.predict_parallel(xs, pool)
            acc = self.accuracy(ys, new_predictions)
            loss = log_loss(ys, new_predictions)
            end = time.time()
            print("Iteration: {}, accuracy: {}, loss: {}, diff: {}, elapsed: {} seconds".format(i + 1, acc, loss, diff, end - start))
        pool.close()
        pool.join()