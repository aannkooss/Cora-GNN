# GNN for Node Classification on Cora Dataset 
This project is an implementation of a basic Graph Neural Network (GNN) using PyTorch to perform node classification on the Cora citation network dataset. The goal is to predict the subject category of a scientific paper based on its content and its relationship to other papers in the network.  

## About the Dataset
The Cora Dataset is a benchmark for GNNs which represents a network of scientific publications where:  
- Nodes (2,708): Each node is a scientific paper
- Edges (10,556): An edge exists between two nodesif one paper cites the other 
- Node Features (1,433): Each paper is described by a binary "bag of words" vector, indicating the presence of absence of 1,433 unique words
- Classes (7): each paper is classified into seven categories 

## Model Architecture 
The model is a simple but effective Graph Convolutional Network with two layers:  
1. GCNConv Layer 1 takes the initial 1433 node features and learns a compact 16 dimensional embedding for each node by aggregating information from its immediate neighbors
2. GCNConv Layer 2 takes the 16 dimensional embedding and produces the final classification output across hte 7 subject categories  
A `ReLU` activation function and `Dropout` are used between layers to learn complex patterns and prevent overfitting  
## How to Run Code 
Before running the code, please ensure that you have a an installation of Python 3.x, PyTorch, and PyTorch Geometric libraries installed  
Installation: 
1. Clone this repository to your machine  
```bash 
git clone https://github.com/aannkooss/Cora-GNN.git
```

2. Create and activate a python virtual environment  
```bash
    python3 -m venv venv
    source venv/bin/activate
```
3. Install the necessary libraries from `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Execution 
1. Once the environment is set up, launch Jupyter Notebook from the root of the project directory. 
```bash 
jupyter notebook
```
