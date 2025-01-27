import zipfile
import os
import matplotlib.image as mpimg
import random
import matplotlib.pyplot as plt

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
