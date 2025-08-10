# **Understanding Backpropagation: Simple Notes**

This document explains the core ideas of how a neural network learns.

## **1\. The Core Components of a Neuron**

Think of a single neuron in the network. It does three simple things.

* **Weights**: These are the **importance knobs** for every connection between neurons.  
  * A large weight (positive or negative) means the connection is very influential.  
  * A weight near zero means the connection is not important.  
  * The network's goal is to "learn" the best settings for these knobs.  
* **Biases**: This is a **starting point** for the neuron. It's an extra number that allows a neuron to activate even if its inputs are zero. It gives the network more flexibility.  
* **Activations**: This is the **final output signal** of a neuron after it does its calculation.  
  1. A neuron receives inputs from the previous layer.  
  2. It calculates a **weighted sum** of those inputs and adds its **bias**.  
  3. This result is put through an **activation function** (like Sigmoid).  
  4. The final number (between 0 and 1 for Sigmoid) is the neuron's "activation" or "firing signal" that it sends to the next layer.

## **2\. The Goal: Learning from Mistakes**

The entire goal of training a neural network is to **minimize error**. We do this in a cycle:

1. **Forward Pass**: Give the network an input (like an image) and let it make a prediction.  
2. **Calculate Error**: Compare the network's prediction to the true answer. The difference is the **error**.  
   * Error \= True\_Answer \- Predicted\_Answer  
3. **Backward Pass (Backpropagation)**: This is where the learning happens. The network travels backward from the error and adjusts its weights and biases to do better next time.  
4. **Repeat**: Do this thousands of times.

## **3\. Backpropagation: The Flow of Blame**

**Backpropagation** is the process of figuring out how much "blame" each weight and bias has for the final error.

Think of the network as a river system. The final error is a flood. Backpropagation is like traveling upstream to see which river branch contributed most to the flood.

The two key rules for distributing this blame are:

1. **Blame is proportional to the connection's weight.** A connection with a large weight had a big influence, so it gets a large share of the blame. A connection with a zero weight gets zero blame.  
2. **Blame is proportional to the neuron's sensitivity.** The derivative of the activation function (like Sigmoid) tells us how "sensitive" a neuron was. A neuron that was on the steep part of its curve was more sensitive and gets more blame.

### **The Math: Variables**
First, let's define the common variables used in the formulas:

- **w**: Weight of a connection.
- **b**: Bias of a neuron.
- **z**: The weighted input to a neuron (`z = w ⋅ a_in + b`). This is the value before the activation function.
- **a**: The activation of a neuron (`a = σ(z)`). This is the final output signal.
- **E**: The Error or Loss function.

### **The Math: Using the Chain Rule**

We use the chain rule from calculus to calculate the exact blame, which we call the **gradient**.

#### **A. Updating Output Weights (Wo​)**

This is the first step backward. We want to find how the error (E) changes when we change the output weights (Wo​).

​`(∂E / ∂Wo) ​ =​(∂E / ∂ypred)​ × (​∂ypred ​ ​/∂zo) × (​∂zo / ∂Wo​​)`

* **Part 1**: Error \= How much we were wrong (`True_Answer - Predicted_Answer`).  
* **Part 2**: Sigmoid Derivative \= How sensitive the final neuron was.  
* **Part 3**: Hidden Layer Output \= The signal that was multiplied by the weight.

#### **B. Updating Hidden Weights (Wh​)**

This is the second step backward. We continue the chain rule.

`(∂E / ∂Wh) ​= ((​∂E / ∂ypred​) ​× (∂ypred​ / ​∂zo​) × (∂zo​​ / ∂Hout​)) × (∂Hout​ / ​∂zh​) × (​∂zh / ​​∂Wh)`

The most important part is in the parenthesis (). This is where we calculate the error for the hidden layer:

`Error_Hidden_Layer = Error_at_Output * Weight_of_Connection`

This is how the error **propagates** (flows) from one layer to the next. We are using the **output weights** (Wo​) to distribute the blame to the hidden layer neurons.

Once we have the `Error_Hidden_Layer`, the rest of the calculation is the same as it was for the output layer.

### **The Final Step: Gradient Descent**

After backpropagation gives us the gradient (the "direction of blame") for every weight, we adjust the weights to reduce the error. We take a small step in the **opposite direction** of the gradient.

`Wnew​ = Wold​ − learning_rate × gradient`

By repeating this process, the weights slowly move toward values that produce the smallest possible error.