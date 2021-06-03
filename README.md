# Blocks World
An implementation of the famous domain of artificial intelligence blocks world.

## Requirements
The program has been tested with python3.7, and the modules required are:
- aima3
- numpy
- cv2
- matplotlib
- keras

In order to use the GUI these other two modules are needed:
- tkinter 
- pillow

## Description
This is just a summarized description of the program, to have a more detailed description you can read the [documentation file](./Relazione/relazione.pdf).
The program recives two input images that represent the initial and the final configuration of the blocks. The two images are processed using the functions in the file `load_state.py`, in order to extract a rappresentation of the states  (as defined in the file `blocks_world.py` using the AIMA module) which can be used by the search algorithms defined in the file `search_algs.py` to output the actions needed to bring the blocks from the initial to the final configurations. In order to recognize the digits that identify the blocks, a convolutional neural networks has been trained on the MNIST dataset, in the file `cnn.py`.

## Usage
There are two ways to use the program:
 - Using the GUI by executing the command `python launch.py`
 - Using che command line script by executing the command `python main.py`. You can use the option `-h` to visualize the helper.

There are some example images that you can use in the folder `images`.
If you want to train the neural network you can do it by launching the command `python cnn.py`, anyway there's a pretrained model in the folder `model`.

