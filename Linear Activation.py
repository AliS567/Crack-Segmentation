#Define the neural network
def get_model(n_ch, patch_height, patch_width):

    inputs = Input(shape=(n_ch,patch_height,patch_width))

    # make it channels_last
    inputs_t = Lambda(lambda x: tf.transpose(x, (0,2,3,1)))(inputs)

    conv1 = Conv2D(32, (3, 3), activation='linear', padding='same')(inputs_t)
    
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='linear', padding='same')(conv1)
    
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='linear', padding='same')(pool1)
    
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='linear', padding='same')(conv2)

    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='linear', padding='same')(pool2)
    
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='linear', padding='same')(conv3)

    up1 = UpSampling2D(size=(2, 2))(conv3)
    up1 = concatenate([conv2,up1],axis=3)

    conv4 = Conv2D(64, (3, 3), activation='linear', padding='same')(up1)
    
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='linear', padding='same')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    up2 = concatenate([conv1,up2], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='linear', padding='same')(up2)
    
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='linear', padding='same')(conv5)

    conv6 = Conv2D(2, (1, 1), activation='linear',padding='same')(conv5)

    # due to channels last, reshape
    conv6 = core.Permute((3,1,2))(conv6)
    conv6 = core.Reshape((2,patch_height*patch_width))(conv6)
    conv6 = core.Permute((2,1))(conv6)
    conv7 = core.Activation('softmax')(conv6)       

    model = Model(input=inputs, output=conv7)

    model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])

    return model


# list of callbacks to include into model.fit
def get_callbacks():
    callbacks = []
    return callbacks