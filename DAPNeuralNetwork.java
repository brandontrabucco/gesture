package com.brandontrabucco.apps.gesture;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Created by Brandon on 10/23/2016.
 */
public class DAPNeuralNetwork {
    private static Random generator;

    private Neuron[][] layers;
    private double learningRate;
    private double decayRate;

    public DAPNeuralNetwork(int[] size, double _learningRate, double _decayRate) {
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

                AssignThreadPool<Neuron[]> pool = new AssignThreadPool<Neuron[]>(1000, layers[i].length, layers[i], activations) {
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

                ReduceThreadPool<Neuron[]> pool = new ReduceThreadPool<Neuron[]>(1000, layers[i].length, layers[i], error) {
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
        public double actionPotential;
        public double actionPotentialPrevious;
        public double actionPotentialPrime;
        public double inductionState;
        public double inductionInput;
        public double inductionInputPrime;
        public double inhibitionFactor;
        public double inhibitionFactorPrevious;
        public double inhibitionFactorPrime;
        public double reductionState;
        public double reductionInput;
        public double reductionInputPrime;

        public double[] inductionWeights;
        public double[] reductionWeights;
        public double[] actionPotentialPreviousPartial;
        public double[] actionPotentialErrorPartial;
        public double[] inductionStatePartial;
        public double[] inductionErrorPartial;
        public double[] inhibitionFactorPreviousPartial;
        public double[] inhibitionFactorErrorPartial;
        public double[] reductionStatePartial;
        public double[] reductionErrorPartial;
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
            actionPotential = 0;
            actionPotentialPrevious = 0;
            actionPotentialPrime = 0;
            inductionState = 0;
            inductionInput = 0;
            inductionInputPrime = 0;
            inhibitionFactor = 0;
            inhibitionFactorPrevious = 0;
            inhibitionFactorPrime = 0;
            reductionState = 0;
            reductionInput = 0;
            reductionInputPrime = 0;

            impulse = new double[connections];
            inductionWeights = new double[connections];
            reductionWeights = new double[connections];
            actionPotentialPreviousPartial = new double[connections];
            actionPotentialErrorPartial = new double[connections];
            inductionStatePartial = new double[connections];
            inductionErrorPartial = new double[connections];
            inhibitionFactorPreviousPartial = new double[connections];
            inhibitionFactorErrorPartial = new double[connections];
            reductionStatePartial = new double[connections];
            reductionErrorPartial = new double[connections];

            for (int i = 0; i < connections; i++) {
                inductionWeights[i] = generator.nextGaussian();
                reductionWeights[i] = generator.nextGaussian();
            }
        }

        public double forward(double[] input) {
            double sum = 0;
            impulse = input.clone();

            // Update the induction input
            for (int i = 0; i < connections; i++) sum += impulse[i] * inductionWeights[i];
            inductionInput = sigmoid(sum);
            inductionInputPrime = inductionInput * (1 - inductionInput);
            sum = 0;

            // Update the reduction input
            for (int i = 0; i < connections; i++) sum += impulse[i] * reductionWeights[i];
            reductionInput = sigmoid(sum);
            reductionInputPrime = reductionInput * (1 - reductionInput);
            sum = 0;

            // Update the induction state
            inductionState = inductionState * inhibitionFactor + inductionInput;
            for (int i = 0; i < connections; i++) {
                actionPotentialPreviousPartial[i] = actionPotentialPrime * (inhibitionFactor * inductionStatePartial[i] + inductionInputPrime * impulse[i]);
                actionPotentialErrorPartial[i] = actionPotentialPrime * (inhibitionFactor * inductionErrorPartial[i] + inductionInputPrime * inductionWeights[i]);
                inductionStatePartial[i] = inductionInputPrime * impulse[i] + inhibitionFactorPrevious * inductionStatePartial[i];
                inductionErrorPartial[i] = inductionInputPrime * inductionWeights[i] + inhibitionFactorPrevious * inductionErrorPartial[i];
            }

            // Update the reduction state
            reductionState = reductionState * (1 - actionPotential) + reductionInput;
            for (int i = 0; i < connections; i++) {
                inhibitionFactorPreviousPartial[i] = inhibitionFactorPrime * ((1 - actionPotential) * reductionStatePartial[i] + reductionInputPrime * impulse[i]);
                inhibitionFactorErrorPartial[i] = inhibitionFactorPrime * ((1 - actionPotential) * reductionErrorPartial[i] + reductionInputPrime * reductionWeights[i]);
                reductionStatePartial[i] = reductionInputPrime * impulse[i] + (1 - actionPotentialPrevious) * inductionStatePartial[i];
                reductionErrorPartial[i] = reductionInputPrime * reductionWeights[i] + (1 - actionPotentialPrevious) * reductionErrorPartial[i];
            }

            // Update the action potential
            actionPotential = sigmoid(inductionState);
            actionPotentialPrime = actionPotential * (1 - actionPotential);

            // Update the inhibition factor
            inhibitionFactor = sigmoid(reductionState);
            inhibitionFactorPrime = inhibitionFactor * (1 - inhibitionFactor);

            // Update the cell activation
            activation = actionPotential * (1 - inhibitionFactor);

            return activation;
        }

        public double[] backward(double errorPrime, double learningRate) {
            double[] weightedError = new double[connections];

            // Pass the error backward through memory cell
            for (int i = 0; i < connections; i++) {
                // Pass the error back through the network, truncating gated error
                weightedError[i] += errorPrime * (1 - inhibitionFactor) * actionPotentialPrime * (inhibitionFactorPrevious * inductionErrorPartial[i] + inductionInputPrime * inductionWeights[i]) + actionPotential * inhibitionFactorPrime * reductionState * actionPotentialErrorPartial[i];
                weightedError[i] += errorPrime * actionPotential * (-1) * inhibitionFactorPrime * ((1 - actionPotentialPrevious) * reductionErrorPartial[i] + reductionInputPrime * reductionWeights[i]) + (1 - inhibitionFactor) * actionPotentialPrime * inductionState * inhibitionFactorErrorPartial[i];

                // Update the induction weights
                inductionWeights[i] -= learningRate * errorPrime * (1 - inhibitionFactor) * actionPotentialPrime * (inhibitionFactorPrevious * inductionStatePartial[i] + inductionInputPrime * impulse[i]) + actionPotential * inhibitionFactorPrime * reductionState * actionPotentialPreviousPartial[i];

                // Update the reduction weights
                reductionWeights[i] -= learningRate * errorPrime * actionPotential * (-1) * inhibitionFactorPrime * ((1 - actionPotentialPrevious) * reductionStatePartial[i] + reductionInputPrime * impulse[i]) + (1 - inhibitionFactor) * actionPotentialPrime * inductionState * inhibitionFactorPreviousPartial[i];
            }

            return weightedError;
        }
    }
}
