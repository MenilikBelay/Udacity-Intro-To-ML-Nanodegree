# By Menilik Belay Woldeyes, Sept 7th, 2024.

import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def process_image (image, image_size: int):
    """
    Processes the given image by converting to Tensor,
    normalizing the values to be between [0 and 1],
    and resizing the image to ({image_size} x {image_size}).
    """
    image = tf.convert_to_tensor(image / 255, dtype=tf.float32)
    return tf.image.resize(image, (image_size, image_size))

def predict (image, model, top_k: int):
    """
    Predicts and returns the top K probable labels for the given image
    by predicting using the given model. The image needs to be of size
    (224, 224, 3).
    """
    
    def list_to_dict (l):
        dic = {}
        for index, val in enumerate (l):
            dic[index] = val
        return dic

    # Convert from (224, 224, 3) to (1, 224, 224, 3)
    image = tf.expand_dims(image, axis=0)
    predictions = model.predict(image)
    label_to_prob = list_to_dict(predictions[0])
    # Capture the keys (labels) by sorting based on the values (probabilities)
    top_labels = sorted(label_to_prob, key=label_to_prob.get, reverse=True)[:top_k]
    return [label_to_prob[label] for label in top_labels], top_labels

def std_print(prediction_prob, prediction_label, top_k, class_names):
    """
    Prints to the console the top K probable labels with their probabilities.
    """
    print (f'Top {top_k} Probabilities')
    for prob, label in zip(prediction_prob, prediction_label):
        print ('{0} : {1}'.format(class_names.get(f'{label+1}'), prob))
    
def plot_bar_graph (image, prediction_prob, prediction_label, top_k, class_names):
    """
    Plots side by side the image and the bar graph for the top K probable labels with their probabilities.
    """
    fig, (ax1, ax2) = plt.subplots(figsize=(10,5), ncols=2)
    ax1.imshow(image)
    ax1.set_title('Flower Image')
    ax2.barh(range(top_k), prediction_prob, height=0.5)
    ax2.set_yticks(range(top_k), [class_names.get(f'{label+1}') for label in prediction_label])
    ax2.set_title('Class Probability')
    plt.tight_layout()
    plt.show()

def parse_command_line_arguments():
    # Usage tutorial: https://pymotw.com/3/argparse/
    parser = argparse.ArgumentParser(
        description='Commandline arguments parser',
    )

    # This is a positional argument, which is mandatory (lacks the - or -- before the flag/argument name).
    parser.add_argument('image_path', action="store", help='Path to the image to be classified')
    parser.add_argument('model_path', action="store", help='Path to the classification model (.h5 file with the extension)')
    # top_k and category_names are optional arguments (starting with - or --).
    parser.add_argument('--top_k', action="store", default=1, required=False, type=int, help='Returns the top K probable classifications')
    parser.add_argument('--category_names', action="store", default='label_map.json', required=False, help='A json file with flower index and labels')
    parser.add_argument('--image_size', action="store", default=224, required=False, type=int, 
                        help='The image size used to train the model')

    return parser.parse_args()

def main():
    """
    Entry point to the program.
    """
    command_line_args = parse_command_line_arguments()

    image = Image.open(command_line_args.image_path)
    image = process_image(np.asarray(image), command_line_args.image_size)
    model = tf.keras.models.load_model(command_line_args.model_path, custom_objects={'KerasLayer':hub.KerasLayer})
    top_k = command_line_args.top_k

    prediction_prob, prediction_label = predict(image, model, top_k)

    with open(command_line_args.category_names, 'r') as f:
        class_names = json.load(f)
    std_print (prediction_prob, prediction_label, top_k, class_names)
    plot_bar_graph(image, prediction_prob, prediction_label, top_k, class_names)

if __name__ == "__main__":
    main()
