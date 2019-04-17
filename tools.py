import cv2
import numpy as np

def draw_line(original_img, predicted_img):
    contours = cv2.findContours(predicted_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    output_img = cv2.drawContours(original_img, contours, -1, (0,0,255), 3)

    return output_img

def overlay(original_img, predicted_img):
    predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_GRAY2BGR)
    predicted_img[np.where((predicted_img == [254, 254, 254]).all(axis = 2))] = [255,0,0]
    output_img = cv2.addWeighted(original_img, 0.7, predicted_img, 0.3, 0)
    return output_img

