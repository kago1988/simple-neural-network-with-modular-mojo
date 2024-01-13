# Define a struct to represent the neural network
struct Network {
  // Number of input nodes
  var inodes: Int

  // Number of hidden nodes in the first layer
  var hnodes_l1: Int

  // Number of hidden nodes in the second layer (optional)
  var hnodes_l2: Int

  // Number of output nodes
  var onodes: Int

  // Learning rate
  var lr: Float32

  // Weights and biases for the first layer
  var wih: Matrix
  var bih: Matrix

  // Weights and biases for the second layer (optional)
  var whh: Matrix
  var bhi: Matrix
  var who: Matrix
  var bho: Matrix
}

# Define a function to initialize the neural network
fn init_network(inout network: Network, input_nodes: Int, hidden_nodes_l1: Int, hidden_nodes_l2: Int, output_nodes: Int, learning_rate: Float32) {
  network.inodes = input_nodes
  network.hnodes_l1 = hidden_nodes_l1
  network.onodes = output_nodes
  network.lr = learning_rate

  // Initialize weights and biases using random values
  network.wih = Matrix.randn(input_nodes, hidden_nodes_l1)
  network.bih = Matrix.zeros(hidden_nodes_l1)

  // Optionally initialize weights and biases for the second layer
  if hidden_nodes_l2 > 0 {
    network.whh = Matrix.randn(hidden_nodes_l1, hidden_nodes_l2)
    network.bhi = Matrix.zeros(hidden_nodes_l2)
    network.who = Matrix.randn(hidden_nodes_l2, output_nodes)
    network.bho = Matrix.zeros(output_nodes)
  }
}

# Define a function to apply the neural network to an input vector
fn apply_network(inout network: Network, input_vector: Vec[Float32]) -> Vec[Float32] {
  // Apply the sigmoid activation function to the input vector
  var z = sigmoid(input_vector)

  // Apply the second layer (if present)
  if network.hnodes_l2 > 0 {
    z = sigmoid(z @ network.whh + network.bhi)
  }

  // Apply the output layer and return the results
  return sigmoid(z @ network.who + network.bho)
}

# Define a function to train the neural network
fn train_network(inout network: Network, inputs: Matrix, targets: Matrix, epochs: Int) {
  for epoch in 0..epochs {
    // Loop through each training example
    for (idx, input_vector) in inputs.each_row() {
      // Apply the neural network to the input vector
      var output_vector = apply_network(network, input_vector)

      // Calculate the error between the predicted output and the target value
      var error = targets[idx] - output_vector[0]

      // Update the weights and biases using the error and the learning rate
      network.wih -= network.lr * error * z @ input_vector
      network.bih -= network.lr * error

      if network.hnodes_l2 > 0 {
        network.whh -= network.lr * error * z @ z
        network.bhi -= network.lr * error
        network.who -= network.lr * error * output_vector @ input_vector
        network.bho -= network.lr * error
      }
    }
  }
}

# Define the `main` function
fn main() {
  // Create some sample data
  var inputs: Matrix = Matrix(Vec::from([1.0, 2.0]), Vec::from([3.0, 4.0]))
  var targets: Matrix = Matrix(Vec::from([0.0]), Vec::from([1.0]))

  // Initialize the neural network
  var network = Network(2, 3, 1, 1, 0.1)

  // Train the neural network
  train_network(&mut network, inputs, targets, 100)
