Key Points of this Architecture:
1. Non-Saturating Non-linearity: 
In this Architecture, we have used ReLu Activation function. Deep Convolution Neural Network with ReLUs(Non Saturating Non-linearity) train 6 times faster than tanh(Saturating Non-linearity).

2. Training on Multiple GPU's:This Architecture has used 2 GTX 580 3 GB GPU's with cross-GPU parallelization scheme. This scheme puts half of the kernels(neurons) on each GPU, with one additional trick: 
the GPUs communicate only in certain layers which results in fast training the model. This scheme reduces top-1 and top-5 error rates by 1.7% and 1.2% respectively.

3. Local Response Normalization:This model has used a neurobiological concept called "Lateral inhibition" for normalizing the input to prevent it from saturating. 
This scheme reduces top-1 and top-5 error rates by 1.4% and 1.2% respectively.

4. Overlapping Pooling:Pooling Layers in CNNs summarize the outputs of neighbouring groups of neurons in the same kernel map. 
We observe that training the models with overlap pooling find it slightly more difficult to overfit. This scheme reduces top-1 and top-5 error rates by 0.4% and 0.3% respectively compared with non-overlapping pooling.

5. Reducing Overfitting by Dropout:Here we have used a popular technique called "Dropout". It consist of setting to zero the output of each neuron with probability 0.5. 
The neurons which are "dropped out" in this way do not contribute to forward pass and do not participate in back propagation. This technique reduces complex co-adaptations of neurons, since a neuron cannot rely on a particular neuron. It is thus forced a neuron to learn more robust features that are useful in conjunction with many different random subsets of other neurons.
We use Dropout in the first two fully-connected layers. Without dropout, our network exhibits substantial overfitting.

