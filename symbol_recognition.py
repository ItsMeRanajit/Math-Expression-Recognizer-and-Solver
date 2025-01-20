# %% [markdown]
# Prediction from images

# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# %% [markdown]
# A. Detecting Symbols from Image

# %%
def symbol_recognition():
    image_dir = "temp/"

    # %%
    import cv2

    def getHorizontalOverlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    def getVerticalOverlap(a, b):
        return max(0, min(a[1], b[1]) - max(a[0], b[0]))

    def detect_contours(img_path, min_distance=50):
        input_image = cv2.imread(img_path, 0) 
        input_image_cpy = input_image.copy()

        binarized = cv2.adaptiveThreshold(input_image_cpy, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        inverted_binary_img = ~binarized

        contours_list, hierarchy = cv2.findContours(
            inverted_binary_img,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        l = []
        for c in contours_list:
            x, y, w, h = cv2.boundingRect(c)
            l.append([x, y, w, h])

        lcopy = l.copy()
        keep = []
        while len(lcopy) != 0:
            curr_x, curr_y, curr_w, curr_h = lcopy.pop(0)
            if curr_w * curr_h < 20:  # Ignore very small contours
                continue
            throw = []
            for i, (x, y, w, h) in enumerate(lcopy):
                curr_interval_x = [curr_x, curr_x + curr_w]
                next_interval_x = [x, x + w]

                curr_interval_y = [curr_y, curr_y + curr_h]
                next_interval_y = [y, y + h]

                # Check both horizontal and vertical overlap
                horizontal_overlap = getHorizontalOverlap(curr_interval_x, next_interval_x)
                vertical_overlap = getVerticalOverlap(curr_interval_y, next_interval_y)

                # Calculate distance between the contours
                distance_x = abs(curr_x - x)
                distance_y = abs(curr_y - y)

                # Merge only if overlaps significantly or are close enough (below min_distance)
                if (horizontal_overlap > min(curr_w, w) * 0.7 and vertical_overlap > min(curr_h, h) * 0.5) or (distance_x < min_distance and distance_y < min_distance):
                    new_interval_x = [min(curr_x, x), max(curr_x + curr_w, x + w)]
                    new_interval_y = [min(curr_y, y), max(curr_y + curr_h, y + h)]
                    newx, neww = new_interval_x[0], new_interval_x[1] - new_interval_x[0]
                    newy, newh = new_interval_y[0], new_interval_y[1] - new_interval_y[0]
                    curr_x, curr_y, curr_w, curr_h = newx, newy, neww, newh
                    throw.append(i)

            # Remove the merged contours from lcopy
            for ind in sorted(throw, reverse=True):
                lcopy.pop(ind)

            # Add the current contour to keep
            keep.append([curr_x, curr_y, curr_w, curr_h])

        return keep


    # %%
    IMAGE = "image_with_border.png"
    img_path = image_dir+IMAGE
    input_image = cv2.imread(img_path, 0) 
    input_image_cpy = input_image.copy()
    keep = detect_contours(image_dir+IMAGE)

    # %%
    for (x, y, w, h) in keep:
        cv2.rectangle(input_image_cpy, (x, y), (x + w, y + h), (0, 0, 255), 1)
    # plt.imshow(input_image_cpy, cmap='gray')
    # plt.show()

    # %% [markdown]
    # B. Predicting Images

    # %%
    def resize_pad(img, size, padColor=255):

        h, w = img.shape[:2]
        sh, sw = size

        if h > sh or w > sw: 
            interp = cv2.INTER_AREA
        else: 
            interp = cv2.INTER_CUBIC

        aspect = w/h  

        if aspect > 1: 
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: 
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: 
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): 
            padColor = [padColor]*3

        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

        return scaled_img

    # %%
    # Load model
    new_model = tf.keras.models.load_model('math-recog-model.h5')
    # new_model = tf.keras.models.load_model('math-recog-model.h5')

    # %%
    def binarize(img):
        img = image.img_to_array(img, dtype='uint8')
        binarized = np.expand_dims(cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2), -1)
        inverted_binary_img = ~binarized
        return inverted_binary_img

    # %%
    class_names = ['(', ')', '+', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'c', 'a', 'b', '/', 'π', '*', 'y']

    # %%
    eqn_list = []
    size_list = False


    input_image = cv2.imread(img_path, 0) 
    inverted_binary_img = binarize(input_image)


    for (x, y, w, h) in sorted(keep, key = lambda x: x[0]):
            if w + h > 50 :
                if w + h < 220 :
                    size_list = True
                else :
                    size_list = False
                
                # print(w,h)

                img = resize_pad(inverted_binary_img[y:y+h, x:x+w], (45, 45), 0)
                
                pred_class = class_names[np.argmax(new_model.predict(tf.expand_dims(tf.expand_dims(img, 0), -1)))]
                # print(pred_class)

                if size_list and pred_class not in {"*", "/", "+", "-","(", "="}:
                    pred_class = "**" + pred_class
                    size_list = False
                if len(eqn_list) > 0 and eqn_list[-1] not in {"*", "/", "+", "-", "(", ")", "="} and pred_class in {'a','b','c','y','π'}:
                    pred_class = "*" + pred_class


                eqn_list.append(pred_class)
    eqn = "".join(eqn_list)
    # print(eqn)
    final_eqn = eqn.replace("**","^")

    # print("Predicted Equation :" ,final_eqn)

    return eqn, final_eqn
