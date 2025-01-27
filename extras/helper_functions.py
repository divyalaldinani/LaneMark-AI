import zipfile
import os
import matplotlib.image as mpimg
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Unzip folder in case the dataset in in the form of Zipped folder
def unzip_folder(folder_path):
    zip_ref = zipfile.ZipFile(folder_path)
    zip_ref.extractall()
    zip_ref.close()


# Check folder structure of a particular folder, mostly dataset folder
def walk_through_directory(dir_path):
    for subdirpath, dirnames, filenames in os.walk(dir_path):
      print(f"{len(dirnames)} directories and {len(filenames)} files in {subdirpath}")

def view_random_image(class_names, directory):
    target_class = random.choice(class_names)
    target_dir = directory + '/' + target_class
    random_image = random.choice(os.listdir(target_dir))
    random_image_path = target_dir + '/' + random_image
    img = mpimg.imread(random_image_path)
    plt.imshow(img)
    plt.title(f"Original class: {target_class}")
    plt.axis(False)


def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = np.arange(len(loss))

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    plt.plot(epochs, loss, label = 'training loss')
    plt.plot(epochs, val_loss, label = 'validation loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()

    plt.plot(epochs, accuracy, label = 'Training accuracy')
    plt.plot(epochs, val_accuracy, label = 'Validation accuracy')
    plt.title('Accuracy curves')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()



def preprocess_image(image_path):
    img = mpimg.imread(image_path)
    img = tf.image.resize(img, (224, 224))
    img /= 255.
    img = tf.expand_dims(img, axis=0)
    return img

def view_predicted_org_and_org_masked_image(directory, model):
    image_name = random.choice(os.listdir(os.path.join(directory, 'frames')))
    image_path = os.path.join(directory, 'frames', image_name)
    image = preprocess_image(image_path)
    plt.figure(figsize=(20, 10))

    
    plt.subplot(1, 3, 1)
    org_image = mpimg.imread(image_path)
    plt.imshow(org_image)
    plt.title('Original Image')
    plt.axis('off')


    plt.subplot(1, 3, 2)
    org_mask_path = os.path.join(directory, 'lane-masks', image_name)
    org_mask = mpimg.imread(org_mask_path)
    plt.imshow(org_mask)
    plt.title('Original Masked Image')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    predicted_mask = model.predict(image)
    predicted_mask = tf.image.resize(predicted_mask, (720, 1280))
    predicted_mask = tf.squeeze(predicted_mask, axis = 0)
    plt.imshow(predicted_mask, cmap='grey')
    plt.title('Predicted Masked Image')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
