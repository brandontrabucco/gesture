package com.brandontrabucco.apps.gesture;

import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by Brandon on 10/23/2016.
 */
public class NeuralNetwork {
    private static Random generator;

    private Neuron[][] layers;
    private double learningRate;
    private double decayRate;

    public NeuralNetwork(int[] size, double _learningRate, double _decayRate) {
        generator = new Random();
        learningRate = _learningRate;
        decayRate = _decayRate;

        // The first layer is full of input neurons and is not added to the network.
        layers = new Neuron[size.length - 1][];
        for (int i = 1; i < size.length; i++) {
            layers[i - 1] = new Neuron[size[i]];
            for (int j = 0; j < size[i]; j++) {
                layers[i - 1][j] = new Neuron(size[i - 1]);
            }
        }
    }

    public double[] forward(double[] input) {
        if (input.length == layers[0][0].connections) {
            double[] activations = input.clone();

            // Iterate forward through the network
            for (int i = 0; i < (layers.length); i++) {
                final double[] connections = activations.clone();
                activations = new double[layers[i].length];

                AssignThreadPool<Neuron[]> pool = new AssignThreadPool<Neuron[]>(100, layers[i].length, layers[i], activations) {
                    @Override
                    public double work(int id, Neuron[] arg) {
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

                ReduceThreadPool<Neuron[]> pool = new ReduceThreadPool<Neuron[]>(100, layers[i].length, layers[i], error) {
                    @Override
                    public double[] work(int id, Neuron[] arg) {
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
}