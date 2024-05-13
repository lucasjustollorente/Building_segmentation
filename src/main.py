from model_utils import define_custom_unet, train_custom_unet, train_pretrained_unet, predict_images, display_images_in_rows
from data_utils import filter_images_by_white_percentage, filter_images_simple, calculate_white_percentage
import os
import argparse
import datetime
import numpy as np
from tqdm import tqdm
os.environ["SM_FRAMEWORK"] = "tf.keras"
#%env SM_FRAMEWORK=tf.keras

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train different types of U-Net models.')
    parser.add_argument('--model_type', type=str, choices=['pretrained', 'from_scratch'], default='pretrained',
                        help='Type of U-Net model to train (pretrained or from_scratch).')
    parser.add_argument('--epochs', type=int, default=75, help='Number of epochs to train.')
    #añado si el preprocesado sera con el porcentaje de pixeles blancos o no
    parser.add_argument('--preprocess', type=str, choices=['white_percentage', 'simple'], default='simple',
                        help='Preprocess method to use for filtering images.')
                    
   

    args = parser.parse_args()

   

    height_shape = 224
    width_shape = 224
    white_threshold = 0.25 
    batch_size = 16
    # Directorios de los conjuntos de datos
    data_path_train = "png/train"
    data_path_train_mask = "png/train_labels"
    data_path_test = "png/test"
    data_path_test_mask = "png/test_labels"
    data_path_val = "png/val"
    data_path_val_mask = "png/val_labels"
        
    if args.preprocess == 'white_percentage':
        X_train, Y_train = filter_images_by_white_percentage(data_path_train, data_path_train_mask, height_shape, width_shape, white_threshold)
        X_test, Y_test = filter_images_by_white_percentage(data_path_test, data_path_test_mask, height_shape, width_shape, white_threshold)
        X_val, Y_val = filter_images_by_white_percentage(data_path_val, data_path_val_mask, height_shape, width_shape, white_threshold)

    else:
        X_train, Y_train = filter_images_simple(data_path_train, data_path_train_mask, height_shape, width_shape)
        X_test, Y_test = filter_images_simple(data_path_test, data_path_test_mask, height_shape, width_shape)
        X_val, Y_val = filter_images_simple(data_path_val, data_path_val_mask, height_shape, width_shape)


    if args.model_type == 'pretrained':
        model, results = train_pretrained_unet(X_train, Y_train, X_test, Y_test, 'resnet50', args.epochs)
        model_name = f'{args.model_type}_unet_{args.epochs}epochs'
        model.save(os.path.join('saved_models', f'{model_name}.h5'))
        model.save_weights(os.path.join('saved_models', f'weights_{model_name}.weights.h5'))

    else:
        model, results = train_custom_unet(X_train, Y_train, X_test, Y_test, height_shape, width_shape, args.epochs)
        model_name = f'{args.model_type}_unet_{args.epochs}epochs'
        model.save(os.path.join('saved_models', f'{model_name}.h5'))
        model.save_weights(os.path.join('saved_models', f'weights_{model_name}.weights.h5'))

    # Predecir las máscaras de las imágenes de validación
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

## python src/main.py --model_type from_scratch --epochs 1 --preprocess simple

    # Print or log training results
    print('Training finished. Model type: {}, Epochs: {}'.format(args.model_type, args.epochs))
    print('Final training accuracy:', results.history['accuracy'][-1])
    print('Final validation accuracy:', results.history['val_accuracy'][-1])
