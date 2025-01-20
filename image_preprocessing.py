# Pre-Processing the Image

import cv2
from matplotlib import pyplot as plt


def image_preprocessing(image_file):
    img = cv2.imread(image_file)

    # %%
    def display(im_path) :
        dpi = 80
        im_data = plt.imread(im_path)

        if(len(im_data.shape) == 2):
            height, width = im_data.shape
        else :
            height, width, depth = im_data.shape


        figsize = width / float(dpi), height / float(dpi)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0,0,1,1])

        ax.axis('off')
        ax.imshow(im_data,cmap='gray')
        plt.show()

    # %%
    # display(image_file)

    # %% [markdown]
    # inverted image

    # %%
    inverted_img=cv2.bitwise_not(img)
    cv2.imwrite("temp/inverted.jpg", inverted_img)


    # %%
    # display("temp/inverted.jpg")

    # %% [markdown]
    # rescaling

    # %% [markdown]
    # binarizatoin

    # %%
    def grayscale(image):
        return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # %%
    gray_img = grayscale(img)
    cv2.imwrite('temp/gray.jpg',gray_img)

    # %%
    # display('temp/gray.jpg')

    # %%
    thresh, im_bw = cv2.threshold(gray_img, 100, 300, cv2.THRESH_BINARY)
    cv2.imwrite('temp/bw.jpg',im_bw)

    # %%
    # display('temp/bw.jpg')

    # %% [markdown]
    # noise removal

    # %%
    def noise_removal(image):
        import numpy as np
        kernel = np.ones((1,1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        kernel = np.ones((1,1),np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image=cv2.morphologyEx(image,cv2.MORPH_CLOSE,kernel)
        # image = cv2.medianBlur(image,1)
        return (image)

    # %%
    no_noise = noise_removal(im_bw)
    cv2.imwrite('temp/no_noise.jpg',no_noise)

    # %%
    # display('temp/no_noise.jpg')

    # %% [markdown]
    # dialation and erosion

    # %%
    import numpy as np
    def thin_font(image):
        
        inverted_image = cv2.bitwise_not(image)
        
        kernel = np.ones((3,5), np.uint8)  
        thinned_image = cv2.erode(inverted_image, kernel, iterations=2)  
        
        thinned_image = cv2.bitwise_not(thinned_image)
        return thinned_image


    # %%
    eroded_img = thin_font(no_noise)
    # eroded_img = no_noise
    cv2.imwrite('temp/eroded.jpg',eroded_img)

    # %%
    # display('temp/eroded.jpg')

    # %%
    def thick_font(image):
        import numpy as np
        image = cv2.bitwise_not(image)
        kernel = np.ones((2,3),np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return (image)

    # %%
    dialated_img = thick_font(eroded_img)
    cv2.imwrite('temp/dialated.jpg', dialated_img)

    # %%
    # display('temp/dialated.jpg')

    # %% [markdown]
    # removing border

    # %%
    def remove_border(image):
        contours, heiarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cntSorted =sorted(contours, key=lambda x:cv2.contourArea(x))
        cnt = cntSorted[-1]
        x,y,w,h = cv2.boundingRect(cnt)
        crop = image[y:y+h, x:x+w]
        return (crop)

    # %%
    no_borders = remove_border(dialated_img)
    cv2.imwrite('temp/no_borders.jpg',no_borders)
    # display('temp/no_borders.jpg')

    # %% [markdown]
    # missing borders

    # %%
    color = [255,255,255]
    top, bottom, left, right = [150]*4

    # %%
    image_with_border = cv2.copyMakeBorder(no_borders, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    cv2.imwrite('temp/image_with_border.png', image_with_border)
    # display('temp/image_with_border.png')


