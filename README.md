## Building Footprints - Segmentación de Imágenes Satelitales

This repository provides tools and scripts for training and using U-Net models to perform satellite image segmentation for identifying building footprints.

## Getting Started

To begin, follow these steps to set up the environment and run the scripts:

1. **Clone the Repository**

   Clone the repository to your local machine:
   
   ```bash
   git clone https://github.com/lucasjustollorente/Building_segmentation.git

2. **Set Up Environment**

   Navigate to the project directory and create a new environment:

   ``python -m venv SATELITE``
   
   ``SATELITE\Scripts\activate``
   
   ``pip install -r requirements.txt``

   Explore the Repository:

   Once the environment is set up, you can explore and use the content within Jupyter Notebook or Visual Studio Code.

  ## Project Structure

   ``png:`` Contains satellite images and their corresponding masks in PNG format, divided into train, train_labels, test, test_labels, val, and val_labels subfolders.

   ``src:`` Contains Python scripts for data manipulation and model training.

   data_utils.py: Utility functions for preprocessing and filtering satellite images.

   model_utils.py: Utility functions for defining and training U-Net models.
          
   main.py: Script to train different types of U-Net models.
          
   predict.py: Script to make predictions using trained models.

   ``notebooks:`` Additional notebooks for testing and experimentation.

   ## Usage
   
   To train a U-Net model:

   --model_type: Type of U-Net model (``pretrained`` or ``from_scratch``). Type of U-Net model to use. Use pretrained for a model that is pretrained using 'resnet50' as the backbone and encoder weights from 'imagenet', or from_scratch to train a U-Net model from scratch.

   --epochs: Number of ``epochs`` to train.

   --preprocess: Preprocessing method for filtering images (``white_percentage`` or ``simple``). Choosing white_percentage involves a more complex preprocessing step that uses a threshold to filter training images with large white areas, potentially improving the final accuracy of the model.

     
     python src/main.py --model_type pretrained --epochs 75 --preprocess white_percentage


   ## Making Predictions
   
   To make predictions using a trained model:

   --model_path: Path to the saved model or weights file.

   --image_path: Path to a single image file for prediction.

   --preprocess: Preprocessing method for image filtering (white_percentage or simple).

    
    python src/predict.py --model_path 'saved_models/pretrained_unet_75epochs.weights.h5' --image_path 'input_image_path/test_1m_San_Diego.tif'



   If you do not specify the image_path the code will run making the predictions with the images in the val path:


   ```
   python src/predict.py --model_path 'saved_models/pretrained_unet_75epochs.weights.h5' --preprocess white_percentage
   ```


   ![image](https://github.com/lucasjustollorente/Building_segmentation/assets/144033748/60e99dff-4625-4b19-bd1b-203ea10560c3)



   ## Notes

   Ensure that your data is organized within the png directory according to the specified structure (train, train_labels, test, test_labels, val, val_labels).

   Adjust parameters such as epochs and model_type according to your requirements.

   For image preprocessing, choose between white_percentage (filtering based on white pixel percentage) or simple (standard filtering).

   Example
   Train a U-Net model from scratch for 50 epochs with simple image preprocessing:

    
    python src/main.py --model_type from_scratch --epochs 50 --preprocess simple.

   ## Results
   Upon completion, the training process will print the final accuracy metrics for both training and validation sets.
   Both the "from_scratch" and "pretrained" models achieve high accuracy on the test set, with over 90% accuracy. The pretrained model performs slightly better because it can more accurately detect finer details such as building edges.

   ## Further Customization
   Feel free to modify the scripts (main.py and predict.py) to add more functionalities or adapt the model architecture based on specific project requirements.

This README provides a guide to setting up and using the repository for satellite image segmentation. Adjust the commands and parameters as needed to fit your use case.
