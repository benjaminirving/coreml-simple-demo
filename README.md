

# Running a Keras machine learning model under macOS with coreML

## Requirements

- Python 2 (unfortunately coremltools only supports Python 2 at the moment)
- The python environment can be reproduced using the Anaconda/miniconda environment.yml file
- macOS 10.13 beta and Xcode 9-beta

## Data

- Example images are included in the `img` folder. Note that these are lower resolution than the original images 
changes the classification results in some cases. 

## Model creation

Run `create_and_save_ml_model.py` to load a pretrained ResNet50 model, 
run it on example images and save it in the mlmodel format

## Swift command line application

- Add the ResNet50.mlmodel to the xcode library
- Build the project
- See example results
- Find the compiled binary and run on an example image e.g. ./CmdSand /path/to/example.img
