import os
num_classes = 20
val_split = 0.2
verbose = 1

# CNN params
batch_size_cnn = 64
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
dense_1_cnn = 512
dense_2_cnn = 256
dense_3_cnn = 128
epochs_cnn = 20
train_dir = os.path.join(os.getcwd(), 'Images/')
cnn_weights = "weights/dog_mobilenet.h5"
cnn_architecture = "weights/dog_mobilenet.json"

## RNN params
csv_path = 'dog_reviews.csv'
vocab_size = 3000
max_length = 30
embedding_dimS = 512
trunc_type = 'post'
oov_tok = "<OOV>"
epochs_rnn = 20
batch_size_rnn = 32
size_lstm  = 128
dense_1_rnn = 256
dense_2_rnn = 64
learning_rate = 0.0001
rnn_weights = "weights/dog_lstm.h5"
rnn_architecture = "weights/dog_lstm.json"