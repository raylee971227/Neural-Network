/*
 * Raymond Lee
 * ECE-469: Artificial Intelligence
 * Neural Net Project
 */

#ifndef NEURAL_NET_NEURALNET_H
#define NEURAL_NET_NEURALNET_H

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <list>
#include <vector>
#include <cmath>

using namespace std;

class NeuralNet {
private:

    class Data{
    public:
        vector<double> inputs;
        vector<double> outputs;
    };

    class DataOut{
    public:
        int a, b, c, d;
        double accuracy, precision, recall, f1;
    };

    class Node;

    class Edge{
    public:
        double weight;
        Node * prev;
        Node * next;
        int p[2];
        int n[2];
    };

    class Node{
    public:
        bool bias;

        vector<Edge *> inEdge;
        vector<Edge *> outEdge;

        double activation;
        double delta;
        double inputSum;
    };

public:
    double sigmoid(double val);
    double sigmoidPrime(double val);
    vector<vector<Node *>> nodes1;
    vector<Edge *> edgeList;

    void train(ifstream &neuralNetFile, ifstream &trainingFile, ofstream &outputFile, int epoch, double learningRate);
    void test(ifstream &neuralNetFile, ifstream &testFile, ofstream &outfile);
    void readNetwork(ifstream &infile);
};


#endif //NEURAL_NET_NEURALNET_H
