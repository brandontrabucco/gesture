package com.brandontrabucco.apps.gesture;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by Brandon on 10/23/2016.
 */
public class LSTMNeuralNetwork {
    private static Random generator;

    private MemoryCell[][] layers;
    private double learningRate;
    private double decayRate;

    public LSTMNeuralNetwork(int[] size, double _learningRate, double _decayRate) {
        generator = new Random();
        learningRate = _learningRate;
        decayRate = _decayRate;

        // The first layer is full of input neurons and is not added to the network.
        layers = new MemoryCell[size.length - 1][];
        for (int i = 1; i < size.length; i++) {
            layers[i - 1] = new MemoryCell[size[i]];
            for (int j = 0; j < size[i]; j++) {
                layers[i - 1][j] = new MemoryCell(size[i - 1]);
            }
        }
    }

    public double[] forward(double[] input) {
        if (input.length + 2 == layers[0][0].connections) {
            double[] activations = input.clone();

            // Iterate forward through the network
            for (int i = 0; i < (layers.length); i++) {
                final double[] connections = activations.clone();
                activations = new double[layers[i].length];

                AssignThreadPool<MemoryCell[]> pool = new AssignThreadPool<MemoryCell[]>(1000, layers[i].length, layers[i], activations) {
                    @Override
                    public double work(int id, MemoryCell[] arg) {
                        return arg[id].forward(connections);
                    }
                };
            }
            return activations;
        }
        else return new double[] {0.0};
    }

    public double[] backward(double[] target) {
        if (target.length == layers[layers.length - 1].length) {
            double[] error = new double[layers[layers.length - 1].length];

            // Calculate the error of the output layer
            for (int i = 0; i < layers[layers.length - 1].length; i++) {
                error[i] = (layers[layers.length - 1][i].activation - target[i]);
            }

            // Iterate through the network in reverse
            for (int i = (layers.length - 1); i >= 0; i--) {
                final double[] weightedError = error.clone();
                error = new double[layers[i][0].connections];

                ReduceThreadPool<MemoryCell[]> pool = new ReduceThreadPool<MemoryCell[]>(1000, layers[i].length, layers[i], error) {
                    @Override
                    public double[] work(int id, MemoryCell[] arg) {
                        return arg[id].backward(weightedError[id], learningRate);
                    }
                };
            }
            learningRate *= decayRate;
            return error;
        } else return new double[]{0.0};
    }

    // A class to perform computations
    private static class WorkerThread implements Runnable {
        public int id;
        public WorkerThread(int id) {
            this.id = id;
        }
        public void run() {
        }
    }

    // A class to manage worker threads for array assignment
    private class AssignThreadPool<T> {
        public AssignThreadPool(int nThreads, int nJobs, final T arg, final double[] output) {
            ExecutorService executor = Executors.newFixedThreadPool(nThreads);
            for (int i = 0; i < nJobs; i++) {
                Runnable worker = new WorkerThread(i) {
                    @Override
                    public void run() {
                        output[id] = work(id, arg);
                    }
                };
                executor.execute(worker); //calling execute method of ExecutorService
            }
            executor.shutdown();
            while (!executor.isTerminated()) {   }

            System.out.println("Finished all threads");
        }

        public double work(int id, T arg) { return 0; }
    }

    // A class to manage worker threads for array reduction
    private class ReduceThreadPool<T> {
        public ReduceThreadPool(int nThreads, int nJobs, final T arg, final double[] output) {
            ExecutorService executor = Executors.newFixedThreadPool(nThreads);
            for (int i = 0; i < nJobs; i++) {
                Runnable worker = new WorkerThread(i) {
                    @Override
                    public void run() {
                        double[] arr = work(id, arg);
                        synchronized (ReduceThreadPool.this) {
                            for (int i = 0; i < arr.length; i++)
                                output[id] += arr[id];
                        }
                    }
                };
                executor.execute(worker); //calling execute method of ExecutorService
            }
            executor.shutdown();
            while (!executor.isTerminated()) {   }

            System.out.println("Finished all threads");
        }

        public double[] work(int id, T arg) { return new double[]{0.0}; }
    }

    // The Neuron class that computes the forward and backward propagation of data
    private static class MemoryCell {
        public double activation;
        public double cellOutput;
        public double cellOutputPrime;
        public double cellInput;
        public double cellInputPrime;
        public double cellState;
        public double outputGate;
        public double outputGatePrime;
        public double forgetGate;
        public double forgetGatePrime;
        public double inputGate;
        public double inputGatePrime;

        public double[] cellInputWeights;
        public double[] outputGateWeights;
        public double[] forgetGateWeights;
        public double[] inputGateWeights;
        public double[] cellInputPartial;
        public double[] cellErrorPartial;
        public double[] forgetGatePartial;
        public double[] inputGatePartial;
        public double[] impulse;
        public int connections;

        private double sigmoid(double input) {
            return 1 / (1 + Math.exp(-input));
        }

        private double sigmoidPrime(double input) {
            return sigmoid(input) * (1 - sigmoid(input));
        }

        private double activate(double input) {
            return Math.tanh(input);
        }

        private double activatePrime(double input) {
            return (1 - (Math.tanh(input) * Math.tanh(input)));
        }

        public MemoryCell(int c) {
            connections = c + 2;
            activation = 0;
            cellOutput = 0;
            cellOutputPrime = 0;
            cellInput = 0;
            cellInputPrime = 0;
            cellState = 0;
            outputGate = 0;
            outputGatePrime = 0;
            forgetGate = 0;
            forgetGatePrime = 0;
            inputGate = 0;
            inputGatePrime = 0;

            impulse = new double[connections];
            cellInputWeights = new double[connections];
            outputGateWeights = new double[connections];
            forgetGateWeights = new double[connections];
            inputGateWeights = new double[connections];
            cellInputPartial = new double[connections];
            cellErrorPartial = new double[connections];
            forgetGatePartial = new double[connections];
            inputGatePartial = new double[connections];

            for (int i = 0; i < connections; i++) {
                cellInputWeights[i] = generator.nextGaussian();
                outputGateWeights[i] = generator.nextGaussian();
                forgetGateWeights[i] = generator.nextGaussian();
                inputGateWeights[i] = generator.nextGaussian();
            }
        }

        public double forward(double[] input) {
            double sum = 0;
            System.arraycopy(new double[]{cellOutput, cellState}, 0, impulse, 0, 2);
            System.arraycopy(input, 0, impulse, 0, input.length);

            // Update the forget gate
            for (int i = 0; i < connections; i++) sum += impulse[i] * forgetGateWeights[i];
            forgetGate = sigmoid(sum);
            forgetGatePrime = forgetGate * (1 - forgetGate);
            for (int i = 0; i < connections; i++) forgetGatePartial[i] = cellState * forgetGatePrime * impulse[i] + forgetGate * forgetGatePartial[i];
            sum = 0;

            // Update the input gate
            for (int i = 0; i < connections; i++) sum += impulse[i] * inputGateWeights[i];
            inputGate = sigmoid(sum);
            inputGatePrime = inputGate * (1 - inputGate);
            for (int i = 0; i < connections; i++) inputGatePartial[i] = cellInput * inputGatePrime * impulse[i] + forgetGate * inputGatePartial[i];
            sum = 0;

            // Update the cell input
            for (int i = 0; i < connections; i++) sum += impulse[i] * cellInputWeights[i];
            cellInput = sigmoid(sum);
            cellInputPrime = cellInput * (1 - cellInput);
            for (int i = 0; i < connections; i++) cellInputPartial[i] = inputGate * cellInputPrime * impulse[i] + forgetGate * cellInputPartial[i];
            sum = 0;

            // Update the cell state
            cellState = cellState * forgetGate + cellInput * inputGate;

            // Update the output gate
            for (int i = 0; i < connections; i++) sum += impulse[i] * outputGateWeights[i];
            outputGate = sigmoid(sum);
            outputGatePrime = outputGate * (1 - outputGate);

            // Update the cell output
            cellOutput = sigmoid(cellState);
            cellOutputPrime = cellOutput * (1 - cellOutput);

            // Update the cell activation
            activation = cellOutput * outputGate;

            return activation;
        }

        public double[] backward(double errorPrime, double learningRate) {
            double[] weightedError = new double[connections];

            // Pass the error backward through memory cell
            for (int i = 0; i < connections; i++) {
                // Pass the error back through the network, truncating gated error
                cellErrorPartial[i] = inputGate * cellInputPrime * cellInputWeights[i] + forgetGate * cellErrorPartial[i];
                weightedError[i] = errorPrime * outputGate * cellOutputPrime * cellErrorPartial[i];

                // Update the output gate weights
                outputGateWeights[i] -= learningRate * errorPrime * cellOutput * outputGatePrime * impulse[i];

                // Update the forget gate weights
                forgetGateWeights[i] -= learningRate * errorPrime * outputGate * cellOutputPrime * forgetGatePartial[i];

                // Update the input gate weights
                inputGateWeights[i] -= learningRate * errorPrime * outputGate * cellOutputPrime * inputGatePartial[i];

                // Update the cell input weights
                cellInputWeights[i] -= learningRate * errorPrime * outputGate * cellOutputPrime * cellInputPartial[i];
            }

            return Arrays.copyOfRange(weightedError, 2, connections);
        }
    }
}