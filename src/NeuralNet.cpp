/*
 * Raymond Lee
 * ECE-469: Artificial Intelligence
 * Neural Net Project
 */

#include "NeuralNet.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

double NeuralNet::sigmoid(double val) {
    return 1.0 / (1.0 + exp(-val));
}

double NeuralNet::sigmoidPrime(double val) {
    return sigmoid(val) * (1- sigmoid(val));
}

void NeuralNet::train(ifstream &neuralNetFile, ifstream &trainingFile, ofstream &outputFile, int epoch, double learningRate) {
    readNetwork(neuralNetFile);
    double nodeVal;
    long long numExamples, numInputs, numOutputs;
    vector<Data> trainingSet;
    Node *node;
    Edge *edge;
    double sum;
    int ctr = 0;

    trainingFile >> numExamples >> numInputs >> numOutputs; //Get neural net attributes
    for(int i = 0; i < numExamples; i++) {
        NeuralNet::Data tmp;

        for(int j = 0; j < numInputs; j++) {
            trainingFile >> nodeVal;
            // Input Node
            tmp.inputs.push_back(nodeVal);
        }

        for(int j = 0; j < numOutputs; j++) {
            trainingFile >> nodeVal;
            // Output Node
            tmp.outputs.push_back(nodeVal);
        }
        trainingSet.push_back(tmp);
    }
    while (ctr < epoch){
        for(int i = 0; i < trainingSet.size(); i++) {

            // Get inputs
            for(int j = 1; j < this->nodes1[0].size(); j++) {
                this->nodes1[0][j]->activation = trainingSet[i].inputs[j-1];
            }

            // Hidden/Output layers
            for(int j = 1; j < this->nodes1.size(); j++) {

                for(int k = 0; k < this->nodes1[j].size(); k++) {
                    if(this->nodes1[j][k]->bias){
                        this->nodes1[j][k]->inputSum = 0.0;
                    }
                    else {
                        sum = 0.0;
                        for(int l = 0; l < this->nodes1[j][k]->inEdge.size(); l++) {
                            sum += this->nodes1[j][k]->inEdge[l]->weight * this->nodes1[j][k]->inEdge[l]->prev->activation;
                        }
                        this->nodes1[j][k]->inputSum = sum;
                        this->nodes1[j][k]->activation = sigmoid(sum);
                    }
                }

            }
            // Get outputs
            for(int j = 0; j < this->nodes1[this->nodes1.size() - 1].size(); j++) {
                node = this->nodes1[this->nodes1.size() - 1][j];
                node->delta = sigmoidPrime(node->inputSum) * (trainingSet[i].outputs[j] - node->activation);
            }

            for(int j = this->nodes1.size() - 2; j > 0; j--) {

                for(int k = 0; k < this->nodes1[j].size(); k++) {
                    node = this->nodes1[j][k];
                    sum = 0.0;

                    for(int l = 0; l < node->outEdge.size(); l++) {
                        sum += node->outEdge[l]->weight * node->outEdge[l]->next->delta;
                    }

                    node->delta = sigmoidPrime(node->inputSum) * sum;
                }

            }

            for(int j = 0; j < this->edgeList.size(); j++) {
                edge = this->edgeList[j];
                edge->weight = edge->weight + (learningRate * edge->prev->activation * edge->next->delta);
            }

        }
        ctr++;
    }

    // Handle file output
    for(int i = 0; i < this->nodes1.size(); i++) {
        outputFile << ((i == this->nodes1.size() - 1) ? this->nodes1[i].size() : this->nodes1[i].size() - 1);
        if(i != this->nodes1.size() - 1) {
            outputFile << " ";
        }
    }

    outputFile << endl;
    for(int i = 1; i < this->nodes1.size(); i++) {

        for(int j = 0; j < this->nodes1[i].size(); j++) {

            if(!this->nodes1[i][j]->bias){
                for(int k = 0; k < this->nodes1[i][j]->inEdge.size(); k++) {
                    outputFile << setprecision(3) << fixed << this->nodes1[i][j]->inEdge[k]->weight;
                    if(k != this->nodes1[i][j]->inEdge.size() - 1){
                        outputFile << " ";
                    }
                }
                outputFile << endl;
            }
        }

    }

}

void NeuralNet::test(ifstream &neuralNetFile, ifstream &testFile, ofstream &outfile) {
    readNetwork(neuralNetFile);
    long long count, numInputs, numOutputs;
    double nodeVal, sum;
    vector<Data> testingSet;
    vector<DataOut> testingOut;
    Data tmp;
    DataOut dataPoint;
    testFile >> count >> numInputs >> numOutputs;

    for(int i = 0; i < count; i++) {

        for(int j = 0; j < numInputs; j++) {
            testFile >> nodeVal;
            tmp.inputs.push_back(nodeVal);
        }

        for(int j = 0; j < numOutputs; j++) {
            testFile >> nodeVal;
            tmp.outputs.push_back(nodeVal);
        }

        testingSet.push_back(tmp);
        tmp.inputs.clear();
        tmp.outputs.clear();

    }

    dataPoint.a = dataPoint.b = dataPoint.c = dataPoint.d = 0;
    for(int i = 0; i < this->nodes1[this->nodes1.size() - 1].size(); i++) {
        testingOut.push_back(dataPoint);
    }

    for(int i = 0; i < testingSet.size(); i++) {

        for(int j = 1; j < this->nodes1[0].size(); j++) {

            this->nodes1[0][j]->activation = testingSet[i].inputs[j - 1];
        }

        for(int j = 1; j < this->nodes1.size(); j++) {

            for(int k = 0; k < this->nodes1[j].size(); k++) {

                if(this->nodes1[j][k]->bias){
                    this->nodes1[j][k]->inputSum = 0.0;
                }
                else {
                    sum = 0.0;
                    for(int l = 0; l < this->nodes1[j][k]->inEdge.size(); l++) {
                        sum += this->nodes1[j][k]->inEdge[l]->weight * this->nodes1[j][k]->inEdge[l]->prev->activation;
                    }
                    this->nodes1[j][k]->inputSum = sum;
                    this->nodes1[j][k]->activation = sigmoid(sum);
                }
            }

        }

        for(int j = 0; j < testingOut.size(); j++) {
            int m = round(this->nodes1[this->nodes1.size() - 1][j]->activation);

            if(m == 1 && testingSet[i].outputs[j] == 1) {
                testingOut[j].a++;
            }
            if(m == 1 && testingSet[i].outputs[j] == 0) {
                testingOut[j].b++;
            }
            if(m == 0 && testingSet[i].outputs[j] == 1) {
                testingOut[j].c++;
            }
            if(m == 0 && testingSet[i].outputs[j] == 0) {
                testingOut[j].d++;
            }
        }
    }

    outfile << setprecision(3) << fixed;
    double a, b, c, d, accuracy, precision, recall, f1;
    for(int i = 0; i < testingOut.size(); i++){
        a = testingOut[i].a;
        b = testingOut[i].b;
        c = testingOut[i].c;
        d = testingOut[i].d;
        testingOut[i].accuracy = (a + d) / (a + b + c + d);
        testingOut[i].precision = a / (a + b);
        testingOut[i].recall = a / (a + c);
        testingOut[i].f1 = (2 * testingOut[i].precision * testingOut[i].recall) / (testingOut[i].precision + testingOut[i].recall);

        outfile << (int)a << " " << (int)b << " " << (int)c << " " << (int)d << " ";
        outfile << testingOut[i].accuracy << " " << testingOut[i].precision << " " << testingOut[i].recall << " " << testingOut[i].f1 << endl;
    }
    a = b = c = d = accuracy = precision = recall = f1 = 0.0;

    for(int i = 0; i < testingOut.size(); i++) {
        a += testingOut[i].a;
        b += testingOut[i].b;
        c += testingOut[i].c;
        d += testingOut[i].d;
    }
    outfile << (a + d) / (a + b + c + d) << " " << a / (a + b) << " " << a / (a + c) << " " << ((2 * (a / (a + b)) * (a / (a + c))) / ((a / (a + b)) + (a / (a + c)))) << endl;

    for(int i = 0; i < testingOut.size(); i++) {
        accuracy += testingOut[i].accuracy;
        precision += testingOut[i].precision;
        recall += testingOut[i].recall;
    }

    accuracy = accuracy / testingOut.size();
    precision = precision / testingOut.size();
    recall = recall / testingOut.size();

    outfile << accuracy << " " << precision << " " << recall << " " << ((2 * precision * recall) / (precision + recall)) << endl;
}

void NeuralNet::readNetwork(ifstream &infile) {
    string buf, token;
    getline(infile, buf);
    istringstream iss(buf);
    vector<int> counts;

    while(iss >> token) {
        counts.push_back(stoul(token, nullptr, 0));
    }

    vector<double> weights;
    vector<vector<double>> nodes;
    vector<vector<vector<double>>> layers;

    weights.push_back(1);
    nodes.push_back(weights);
    weights.clear();

    // Read input
    for(int i = 0; i < counts[0]; i++){
        weights.push_back(1);
        nodes.push_back(weights);
        weights.clear();
    }

    layers.push_back(nodes);
    nodes.clear();

    for(int i = 1; i < counts.size(); i++) {
        if(i < counts.size() - 1) {
            weights.push_back(1);
            nodes.push_back(weights);
            weights.clear();
        }

        for(int j = 0; j < counts[i]; j++) {
            getline(infile, buf);
            iss = istringstream(buf);
            while(iss >> token) {
                weights.push_back(strtod(token.c_str(), nullptr));
            }
            nodes.push_back(weights);
            weights.clear();
        }
        layers.push_back(nodes);
        nodes.clear();
    }
    // Set up edge weights
    Node* node;
    Edge * edge;
    vector<Node *> layerNodes;

    for(int i = 0; i < layers[0].size(); i++) {
        node = new Node;

        if(i) {
            node->activation = 0.0;
            node->bias = false;
        }
        else {
            node->activation = -1.0;
            node->bias = true;
        }
        node->inEdge.push_back(nullptr);
        layerNodes.push_back(node);
    }
    this->nodes1.push_back(layerNodes);
    layerNodes.clear();

    // Config hidden layer
    for(int i = 1; i < layers.size() - 1; i++) {
        for(int j = 0; j < layers[i].size(); j++) {
            node = new Node;
            if(j){
                node->activation = 0.0;
                node->bias = false;
                for(int k = 0; k < layers[i][j].size(); k++) {
                    edge = new Edge;
                    edge->weight = layers[i][j][k];
                    edge->prev = this->nodes1[i-1][k];
                    edge->next = node;
                    edge->p[0] = i - 1;
                    edge->p[1] = k;
                    edge->n[0] = i;
                    edge->n[1] = j;
                    this->edgeList.push_back(edge);
                    this->nodes1[i-1][k]->outEdge.push_back(edge);
                    node->inEdge.push_back(edge);
                }
            }
            else {
                node->activation = -1.0;
                node->bias = true;
                node->inEdge.push_back(nullptr);
            }
            layerNodes.push_back(node);
        }
        this->nodes1.push_back(layerNodes);
        layerNodes.clear();
    }

    // Config output layer
    for(int i = 0; i < layers[layers.size()-1].size(); i++) {
        node = new Node;
        node->activation = 0.0;
        node->bias = false;

        for(int j = 0; j < layers[layers.size()-1][i].size(); j++) {
            edge = new Edge;
            edge->weight = layers[layers.size() - 1][i][j];
            edge->prev = this->nodes1[layers.size() - 2][j];
            edge->next = node;
            edge->p[0] = layers.size() - 2;
            edge->p[1] = j;
            edge->n[0] = layers.size() - 1;
            edge->n[1] = i;
            this->edgeList.push_back(edge);
            this->nodes1[layers.size() - 2][j]->outEdge.push_back(edge);
            node->inEdge.push_back(edge);
        }

        layerNodes.push_back(node);
    }

    this->nodes1.push_back(layerNodes);
    layerNodes.clear();
}