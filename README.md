

# Running a Keras machine learning model under macOS with coreML

[see blog post](http://www.birving.com/blog/2017/09/02/the-move-towards-library-agnostic-machine-learning)

## Updates:
- Code updated for Python 3

## Requirements

- A python virtual environment can be created from the requirements file. 
- macOS

## Data

- Example images are included in the `img` folder. Note that these are lower resolution than the original images 
changes the classification results in some cases. 

## Model creation

Run `create_and_save_ml_model.py` to load a pretrained ResNet50 model, 
run it on example images and save it in the mlmodel format

## Swift command line application

- Add the ResNet50.mlmodel to the Xcode library
- Build the project in Xcode
- See example results in blog
- Find the compiled binary and run on an example image e.g. `./CmdSand /path/to/example.img`
