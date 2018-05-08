
# coding: utf-8

# In[1]:


from keras import Model
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Dense, Flatten, Dropout
from keras.applications.imagenet_utils import _obtain_input_shape


# In[2]:


def Rapid_Net(shape_img, num_classes):
    input_shape= Input(shape = shape_img)
    pool_1 = AveragePooling2D(pool_size = (2,2))(input_shape)
    conv_1 = Conv2D(32,kernel_size = (5,5), activation= 'relu')(pool_1)

    pool_2 = MaxPooling2D(pool_size = (2,2))(conv_1)
    conv_2 = Conv2D(32,kernel_size = (3,3), activation = 'relu')(pool_2)


    #drop_1 = Dropout(0.25)(pool_1)

    flatten_1 = Flatten()(conv_2)

    dense_1 = Dense(1024, activation='relu')(flatten_1)
    #dense_2 = Dense(1024, activation='relu')(dense_1)

    out = Dense(num_classes, activation='softmax')(dense_1)
    model = Model(input_shape, out, name='RapidNet')
    return model
