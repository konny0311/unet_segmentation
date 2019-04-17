import numpy as np
import glob
import os
import cv2
import shutil
from keras.engine.topology import Input
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import  Conv1D, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.layers.normalization import BatchNormalization
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import plot_model
import keras.callbacks as KC
from history_checkpoint_callback import HistoryCheckpoint, TargetHistoryi

NUM_CLASSES = 1
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
TEST_IMG_DIR = 'images/test/'
BEST_MODEL_PATH = 'models/best_model_weights.hdf5'
COMP_MODEL_PATH = 'models/comp_model_weights.hdf5'
PRED_DIR = 'images/pred/'

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

def create_images_answers(dir_path, teacher=False, filename=False):
    files = glob.glob(os.path.join(dir_path, '*.png'))
    files.sort()
    if teacher:
        COLOR_MODE = 0
    else:
        COLOR_MODE = 1
    images = [resize_for_model(cv2.imread(file, COLOR_MODE)) for file in files]
    if filename:
        return images, files
    else:
        return images

def prepare_data():
    train_images = create_images_answers(TRAIN_IMG_DIR)
    train_teacher = create_images_answers(TRAIN_ANNOT_DIR, teacher=True)
    valid_images = create_images_answers(VALID_IMG_DIR)
    valid_teacher = create_images_answers(VALID_ANNOT_DIR, teacher=True)
    test_images, test_filenames = create_images_answers(TEST_IMG_DIR, filename=True)

    train_images = np.array(train_images) / 255
    train_teacher = np.array(train_teacher) / 255
    valid_images = np.array(valid_images) / 255
    valid_teacher = np.array(valid_teacher) / 255
    test_images = np.array(test_images) / 255

    t_shape = train_teacher.shape
    v_shape = valid_teacher.shape
    train_teacher = train_teacher.reshape(t_shape[0], t_shape[1], t_shape[2], 1)
    valid_teacher = valid_teacher.reshape(v_shape[0], v_shape[1], v_shape[2], 1)

    return train_images, train_teacher, valid_images, valid_teacher, test_images, test_filenames

def create_callbacks():
    callbacks = [KC.TensorBoard(),
                HistoryCheckpoint(filepath='chart/LearningCurve_{history}.png'
                                , verbose=1
                                , period=2
                                , targets=[TargetHistory.Loss, TargetHistory.DiceCoef, TargetHistory.ValidationLoss, TargetHistory.ValidationDiceCoef]
                               ),
                KC.ModelCheckpoint(filepath=BEST_MODEL_PATH,
                verbose=1,
                save_weights_only=True, #model全体を保存
                save_best_only=True,
                period=10)]

    return callbacks

def predict_output(model, test_images, test_filenames):
    pred_images = model.predict(test_images)
    for j, image in enumerate(pred_images):
        name = test_filenames[j].split('/')[-1]
        cv2.imwrite(PRED_DIR + name, image)
    shutil.make_archive('pred_images', 'zip', root_dir = PRED_DIR)

def main():
    train, train_teacher, valid, valid_teacher, test_images, test_filenames = prepare_data()
    print('shape of train', train.shape)
    print('shape of train teacher', train_teacher.shape)
    print('shape of valid', valid.shape)
    print('shape of valid teacher', valid_teacher.shape)

    shape = (train.shape[1], train.shape[2], train.shape[3])
    model = create_unet_model(shape)
    model.compile(loss=dice_loss, optimizer=Adam(lr=LEARNING_RATE), metrics=[dice_coef])
    model.summary()
    # plot_model(model)
    callbacks = create_callbacks()
    history = model.fit(train, train_teacher, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, validation_data=(valid, valid_teacher), shuffle=True, callbacks=callbacks)

    model.save_weights(COMP_MODEL_PATH)
    predict_output(model, test_images, test_filenames)

if __name__ == '__main__':
    main()
