/*
 * Raymond Lee
 * ECE-469: Artificial Intelligence
 * Neural Net Project
 */

#include <iostream>
#include <fstream>
#include <string>

#include "NeuralNet.h"

using namespace std;

int train() {
    string initNeuralNet, outputFile, trainingFile, epochs;
    string::size_type sz;
    int epoch;
    double learningRate;
    ifstream initfile, trainfile;
    ofstream outfile;

    cout << "Please input name an initial neural network file: ";
    cin >> initNeuralNet;

    initfile.open(initNeuralNet.c_str());
    // Error checking
    if(!initfile) {
        cerr << "error: could not open " << initNeuralNet << endl;
        return(-1);
    }

    cout << "Please input a training set in which to train the neural network: ";
    cin >> trainingFile;

    trainfile.open(trainingFile.c_str());
    if(!trainfile) {
        cerr << "error: could not open " << trainingFile << endl;
        return(-1);
    }

    cout << "Please input a file in which the output the trained neural network: ";
    cin >> outputFile;

    outfile.open(outputFile.c_str());
    if(!outfile) {
        cerr << "error: could not open " << outputFile << endl;
        return(-1);
    }

    cout << "Please specify the number of epochs: ";
    cin >> epochs;
    epoch = stoi(epochs);

    cout << "Please specify a learning rate: ";
    cin >> learningRate;

    NeuralNet network;
    network.train(initfile, trainfile, outfile, epoch, learningRate);

    initfile.close();
    trainfile.close();
    outfile.close();
    return 0;
}

int test() {

    string neuralNet, outputFile, test;
    ifstream neuralNetFile, testFile;
    ofstream resultsFile;

    cout << "Please input a neural network to test: ";
    cin >> neuralNet;

    neuralNetFile.open(neuralNet.c_str());
    if(!neuralNetFile) {
        cerr << "error: could not open " << neuralNet << endl;
        return(-1);
    }

    cout << "Please input a test set to test neural network: ";
    cin >> test;

    testFile.open(test.c_str());
    if(!testFile) {
        cerr << "error: could not open " << test << endl;
        return(-1);
    }

    cout << "Please input a results file: ";
    cin >> outputFile;

    resultsFile.open(outputFile.c_str());
    if(!resultsFile) {
        cerr << "error: could not open " << outputFile << endl;
        return(-1);
    }

    NeuralNet network;
    network.test(neuralNetFile, testFile, resultsFile);

    neuralNetFile.close();
    testFile.close();
    resultsFile.close();

    return 0;
}

int main() {
    int mode;

    cout << "Please select a mode to run" << endl;
    cout << "Press [1] to train a neural network" << endl;
    cout << "Press [2] to test a neural network" << endl;
    cin >> mode;

    if(mode != 1 && mode != 2) {
        cout << "Please select a valid mode." << endl;
        cin >> mode;
    }

    if(mode == 1) {
        train();
    }
    else if(mode == 2) {
        test();
    }

    return 0;
}