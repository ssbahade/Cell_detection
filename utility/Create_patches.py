import numpy as np
from numpy.lib.stride_tricks import as_strided
import cv2
from skimage.transform import resize
import tensorflow as tf
# image_shape = (224,224)

'''
img = np.arange(6500 * 4500, dtype=np.float32).reshape(6500, 4500)
img_strided = as_strided(img, shape=(6500 - 3000, 4500 - 3000, 31, 31),
                         strides=img.strides + img.strides, writeable=False)
# img_strided.shape == (6470, 4470, 31, 31)
for img_patch in img_strided.reshape(-1, 31, 31):
#     img_patch.shape == (patch_height, patch_width) '''
#
# patch_size = 64
# ws2 = int(patch_size/2)
#image_inp = cv2.imread('D:/PhD_Image_Data/PhD project/PycharmProjects/2D/data/vgg/Training_data/Images/TMA1_A1.tif')
#image_inp = np.reshape(image_inp,[-1,768,1366,3])
# dimensions = image.shape
# height = dimensions[0]
# width = dimensions[1]
'''
imtest = np.ndarray(shape=(int(width-2*ws2), 1, patch_size, patch_size,3), dtype=float)


for y in range(ws2,width-ws2):
    for x in range(ws2,height-ws2):
        imtest[x-ws2,0] = image[y-ws2:y+ws2, x-ws2:x+ws2]


patch_height=31
patch_width=31

import numpy as np
from numpy.lib.stride_tricks import as_strided

img = np.arange(768 * 1366, dtype=np.float32).reshape(768, 1366)
img_strided = as_strided(img, shape=(768 - 30, 1366 - 30, 31, 31),
                         strides=img.strides + img.strides, writeable=False)
# img_strided.shape == (6470, 4470, 31, 31)
for img_patch in img_strided.reshape(-1, 31, 31):
    img_patch.shape == (patch_height, patch_width)
'''
# windowSize = (31, 31)
# stepSize = 32
# Img_window = []
def sliding_window(image, stepSize, windowSize):
    for y in range(0,image.shape[0],stepSize):
        for x in range(0, image.shape[1],stepSize):
            yield (x,y,image[y:y + windowSize[1], x:x + windowSize[0]])

#
# for (x, y, window) in sliding_window(image, stepSize, windowSize):
#     window = resize(window, image_shape)
#     Img_window.append(window)


#patches=tf.image.extract_image_patches(image_inp,ksizes=[1, 224, 224, 1],strides=[1, 112, 112, 1],rates=[1, 1, 1, 1],padding="SAME")



def remove_unsized_patched(data_folder):
    ####################generator function to load images and Label having 664 patches and each having size (224,224,3)
    import numpy as np
    from Preprocessing import all_image_into_patches



    #################### store 664 images and label into img and gt_l
    im,gt = all_image_into_patches(data_folder)
    img = np.asarray(im)
    gt_l = np.asarray(gt)
    print("img type and shape:",type(img),img.shape)           # img[21] = [93,224,3]   delete thses images
    print("gtl type and shape:",type(gt_l),gt_l.shape)         # gt_l[21] = [93,224,3]  delete theses images
    print("After deleting unsized images:")

    ############################# remove unshaped image  like (64,224,3) or (93,16,3) or removed unshaped patches
    c = 0
    list_correct_sized_image = []
    for s in range(0,len(img)):
        if img[s].shape == (224, 224,3):
            list_correct_sized_image.append(img[s])
            c = c+1
    print("counter n0. of sized images:",c,np.shape(list_correct_sized_image), type(list_correct_sized_image))

    # remove unshaped label  like (64,224,3) or (93,16,3) or removed unshaped patches
    d=0
    list_correct_sized_gt = []
    for s in range(0,len(gt_l)):
        if gt_l[s].shape == (224, 224):
            list_correct_sized_gt.append(gt_l[s])
            d = d+1
    print("counter n0. of sized images:",d,np.shape(list_correct_sized_gt), type(list_correct_sized_gt))
    return np.array(list_correct_sized_image),np.array(list_correct_sized_gt)






#remove_unsized()

'''
temp_img = np.ndarray([len(img), 224, 224, 3])
for i in range(0,len(img)):
    temp_img[i]=img[i]

print(("temp_img.shape:",temp_img.shape))
'''
# image_4d_array_list = []
# gt_4d_array_list = []
#
# for shape in range(0,len(img)):
#     #image_4d_array=np.reshape(img[shape],[-1,224,224,3])
#     image_4d_array_list.append(img[shape])
#     image_4d_array_temp = np.reshape(image_4d_array_list, [-1, 224, 224, 3])
#     print("4d shape im: ",image_4d_array_temp.shape)
#
#     #gt_4d_array=np.reshape(gt_l[shape],[-1,224,224,3])
#     gt_4d_array_list.append(gt_l[shape])

#
# print("image_4d_array_list type:{} and shape:{}".format(type(image_4d_array_list),np.shape(image_4d_array_list)))
# print("gt_4d_array_list type:{} and shape:{}".format(type(gt_4d_array_list),np.shape(gt_4d_array_list)))


#print("img.shape", type(a))
#print("gt.shape", np.shape(a))








