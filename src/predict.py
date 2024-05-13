import os
import argparse
import numpy as np
from model_utils import  predict_images, display_images_in_rows, display_resized_image_and_mask
from data_utils import filter_images_by_white_percentage, filter_images_simple, calculate_white_percentage, calculate_white_percentage, preprocess_image
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras.models import load_model
import tensorflow as tf
import segmentation_models as sm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict using a trained U-Net model.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model or weights file.')
    parser.add_argument('--image_path', type=str,
                        help='Path to the image file for prediction.')
    parser.add_argument('--preprocess', type=str, choices=['white_percentage', 'simple'], default='simple',
                        help='Preprocess method to use for filtering images.')
    args = parser.parse_args()

    height_shape = 224
    width_shape = 224
    white_threshold = 0.25 
    # Directorios de los conjuntos de datos
    try:
        # Load or define model for prediction
        model = load_model(args.model_path)
    except:
        BACKBONE = 'resnet50'
        model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
        model.load_weights(args.model_path)

    if args.image_path:
        # Process single specified image
        input_image = preprocess_image(args.image_path, height_shape, width_shape)
        try:
            predicted_mask = model.predict(input_image)
            images_rows = [[input_image, predicted_mask]]
            titles_rows = [["Input Image", "Predicted Mask"]]

        except:
            input_image= np.expand_dims(input_image, axis=0)
            predicted_mask = model.predict(input_image)
            images_rows = [[input_image, predicted_mask]]
            titles_rows = [["Input Image", "Predicted Mask"]]
        
        display_resized_image_and_mask(input_image, predicted_mask)
        

    
    else:

        data_path_val = "png/val"
        data_path_val_mask = "png/val_labels"
            
        if args.preprocess == 'white_percentage':
            X_val, Y_val = filter_images_by_white_percentage(data_path_val, data_path_val_mask, height_shape, width_shape, white_threshold)

        else:
            X_val, Y_val = filter_images_simple(data_path_val, data_path_val_mask, height_shape, width_shape)
        # Preprocess validation images

        # Predict masks for validation images
        Y_val_pred = predict_images(model, X_val)

        # Prepare images and titles for visualization
        images_rows = []
        titles_rows = []
        for idx in range(len(X_val)):
            input_image = X_val[idx]
            ground_truth_mask = np.squeeze(Y_val[idx])
            predicted_mask = np.squeeze(Y_val_pred[idx])
            images_row = [input_image, ground_truth_mask, predicted_mask]
            titles_row = ["Input Image", "Ground Truth Mask", "Predicted Mask"]
            images_rows.append(images_row)
            titles_rows.append(titles_row)

        # Display the images in rows of three
        display_images_in_rows(images_rows, titles_rows)

