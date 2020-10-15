import os
batch_size = 64
valid_size = 32
color_mode = 'rgb'
width = 224
height = 224
target_size = (width, height)
input_shape = (width, height, 3)
shear_range = 0.2
zoom_range = 0.15
rotation_range = 20
shift_range = 0.2
rescale = 1./255
dense_1 = 512
dense_2 = 256
dense_3 = 128
dense_4 = 64
num_classes = 120
epochs = 10
verbose = 1
val_split = 0.15

# data directories and model paths
train_dir = os.path.join(os.getcwd(), 'Images/')
model_weights = "weights/dog_mobilenet.h5"
model_architecture = "weights/dog_mobilenet.json"