package com.brandontrabucco.apps.gesture;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by Brandon on 10/23/2016.
 */
public class ConvolutionalNeuralNetwork {
    private static Random generator;

    private Filter[][] filters;
    private Neuron[][] layers;
    private double learningRate;
    private double decayRate;

    public ConvolutionalNeuralNetwork(int filterSize, int inputWidth, int inputHeight, int[] filterCount, int[] neuronCount, double _learningRate, double _decayRate) {
        generator = new Random();
        learningRate = _learningRate;
        decayRate = _decayRate;

        filters = new Filter[filterCount.length][];
        for (int i = 0; i < filterCount.length; i++) {
            filters[i] = new Filter[filterCount[i]];
            for (int j = 0; j < filterCount[i]; j++) {
                if (i != 0) filters[i][j] = new Filter(filterSize, filters[i - 1][0].activation.length, filters[i - 1][0].activation[0].length);
                else  filters[i][j] = new Filter(filterSize, inputWidth, inputHeight);
            }
        }

        layers = new Neuron[neuronCount.length][];
        for (int i = 0; i < neuronCount.length; i++) {
            layers[i] = new Neuron[neuronCount[i]];
            for (int j = 0; j < neuronCount[i]; j++) {
                if (i != 0) layers[i][j] = new Neuron(neuronCount[i - 1]);
                else layers[i][j] = new Neuron(filters[filters.length - 1].length * filters[filters.length - 1][0].activation.length * filters[filters.length - 1][0].activation[0].length);
            }
        }
    }

    public double[] forward(double[][] input) {
        double[][] activations = input.clone();

        // Iterate forward through the network
        for (int i = 0; i < (filters.length); i++) {
            for (int j = 0; j < (filters[i].length); j++) {
                final double[][] connections = activations.clone();
                activations = filters[i][j].convolve(connections);
            }
        }

        double[] output = new double[activations.length * activations[0].length];
        for (int i = 0; i < activations.length; i++) for (int j = 0; j < activations[0].length; j++) output[i * activations[0].length + j] = activations[i][j];

        // Iterate forward through the network
        for (int i = 0; i < (layers.length); i++) {
            final double[] connections = output.clone();
            output = new double[layers[i].length];

            AssignThreadPool<Neuron[]> pool = new AssignThreadPool<Neuron[]>(100, layers[i].length, layers[i], output) {
                @Override
                public double work(int id, Neuron[] arg) {
                    return arg[id].forward(connections);
                }
            };
        }
        return output;
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
    private static class Neuron {
        public double activation;
        public double activationPrime;
        public double[] weights;
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

        public Neuron(int c) {

            connections = c;
            activation = 0;
            activationPrime = 0;

            impulse = new double[c];
            weights = new double[c];
            for (int i = 0; i < c; i++) {
                weights[i] = generator.nextGaussian();
            }
        }

        public double forward(double[] input) {
            double sum = 0;
            impulse = input.clone();

            // find the weighted sum of all input
            for (int i = 0; i < connections; i++) sum += input[i] * weights[i];
            activation = activate(sum);
            activationPrime = activatePrime(sum);
            return activation;
        }

        public double[] backward(double errorPrime, double learningRate) {
            double[] weightedError = new double[connections];

            // update all weights
            for (int i = 0; i < connections; i++) {
                weightedError[i] = (errorPrime * weights[i] * activationPrime);
                weights[i] -= learningRate * errorPrime * impulse[i] * activationPrime;
            }
            return weightedError;
        }
    }

    // The Filter class that computes convolutions of an image
    private static class Filter {
        public double[][] activation;
        public double[][] activationPrime;
        public double[][] weights;
        public double[][] impulse;
        public int filterSize;
        public int inputWidth;
        public int inputHeight;

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

        public Filter(int _filterSize, int _inputWidth, int _inputHeight) {

            inputWidth = _inputWidth;
            inputHeight = _inputHeight;
            filterSize = _filterSize;
            activation = new double[_inputWidth - _filterSize][_inputHeight - _filterSize];
            activationPrime = new double[_inputWidth - _filterSize][_inputHeight - _filterSize];

            impulse = new double[_inputWidth][_inputHeight];
            weights = new double[_filterSize][_filterSize];
            for (int i = 0; i < _filterSize; i++) for (int j = 0; j < _filterSize; j++) {
                weights[i][j] = generator.nextGaussian();
            }
        }

        public double[][] convolve(double[][] input) {
            impulse = input.clone();

            // iterate through the entire image, given filter size
            for (int i = 0; i < inputWidth - filterSize; i++) {
                for (int j = 0; j < inputHeight - filterSize; j++) {
                    // iterate through each weight in the filter
                    double sum = 0;
                    for (int k = 0; k < filterSize; k++) {
                        for (int l = 0; l < filterSize; l++) {
                            // multiply each filter weight by pixel
                            sum += input[i + k][j + l] * weights[k][l];
                        }
                    }
                    // pass the sum through a sigmoid
                    activation[i][j] = sigmoid(sum);
                    activationPrime[i][j] = sigmoidPrime(sum);
                }
            }

            return activation;
        }
    }
}