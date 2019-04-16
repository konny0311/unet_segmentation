import numpy as np
import glob
import os
import cv2
from keras.engine.topology import Input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import plot_model

# SHAPE = (128, 128, 3)
NUM_CLASSES = 2
LEARNING_RATE = 0.001
COLOR_MODE = 1
SIZE = 128
BATCH_SIZE = 40
EPOCHS = 30
VERBOSE = 1
TRAIN_IMG_DIR = 'images/train/original/'
TRAIN_ANNOT_DIR = 'images/train/annot/'
VALID_IMG_DIR = 'images/valid/original/'
VALID_ANNOT_DIR = 'images/valid/annot/'

def conv_part(input_layer, filter=64, padding_same = True, first_layer = False):
    """
    max pool 2x2 -> conv 3x3(relu) -> conv 3x3(relu)(as an output_layer)
    """
    if padding_same:
        pad_type = 'same'
    else:
        pad_type = 'valid'
    
    if first_layer:
        layer1 = input_layer
    else:
        layer1 = MaxPooling2D()(input_layer)
    layer2 = Conv2D(filter, 3, activation='relu', padding=pad_type)(layer1)
    output_layer = Conv2D(filter, 3, activation='relu', padding=pad_type)(layer2)

    return output_layer

def deconv_part(input_layer, copied_layer, filter=64, padding_same = True):
    """
    upsampling 2x2 -> conv 3x3 -> concat -> conv 3x3(relu) -> conv 3x3(relu)(as an output_layer)
    """
    if padding_same:
        pad_type = 'same'
    else:
        pad_type = 'valid'

    upsample_layer = UpSampling2D()(input_layer)
    layer1 = Conv2D(filter, 3, activation='relu', padding=pad_type)(upsample_layer)
    layer1 = concatenate([layer1, copied_layer],axis=3)
    layer2 = Conv2D(filter, 3, activation='relu', padding=pad_type)(layer1)
    output_layer = Conv2D(filter, 3, activation='relu', padding=pad_type)(layer2)

    return output_layer

def create_unet_model(shape):

    input_layer = Input(shape=shape)
    en_layer1 = conv_part(input_layer, filter=64, first_layer=True)
    en_layer2 = conv_part(en_layer1, filter=128)
    en_layer3 = conv_part(en_layer2, filter=256)
    en_layer4 = conv_part(en_layer3, filter=512)
    de_layer3 = deconv_part(en_layer4, en_layer3, filter=256)
    de_layer2 = deconv_part(de_layer3, en_layer2, filter=128)
    de_layer1 = deconv_part(de_layer2, en_layer1, filter=64)
    output = Conv2D(NUM_CLASSES, 1, padding="same", activation='sigmoid')(de_layer1)
    model = Model(inputs=[input_layer], outputs=[output])

    return model

def dice_coef(y_true, y_pred):
    """
    自作損失関数ダイス係数 https://mathwords.net/jaccardkeisu
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)

    return 2 * intersection / (K.sum(y_true) + K.sum(y_pred))

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def resize_for_model(image):
    # np形式のimageを特定の大きさにresizeする。
    return cv2.resize(image, (SIZE, SIZE))

def create_images_answers(dir_path):
    files = glob.glob(os.path.join(dir_path, '*.png'))
    print(dir_path, len(files))
    files.sort()
    images = [resize_for_model(cv2.imread(file, COLOR_MODE)) for file in files]

    return images    

def prepare_data():
    train_images = create_images_answers(TRAIN_IMG_DIR)
    train_annot = create_images_answers(TRAIN_ANNOT_DIR)
    valid_images = create_images_answers(VALID_IMG_DIR)
    valid_annot = create_images_answers(VALID_ANNOT_DIR)

    return np.array(train_images), np.array(train_annot), np.array(valid_images), np.array(valid_annot)


def main():
    train, train_teacher, valid, valid_teacher = prepare_data()
    print('shape of train', train.shape)
    shape = (train.shape[1], train.shape[2], train.shape[3])
    model = create_unet_model(shape)
    model.compile(loss=dice_loss, optimizer=Adam(lr=LEARNING_RATE), metrics=[dice_coef])
    model.summary()
    # plot_model(model)
    history = model.fit(train, train_teacher, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_data=(valid, valid_teacher), shuffle=True)


if __name__ == '__main__':
    main()