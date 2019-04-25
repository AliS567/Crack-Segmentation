#Define the neural network
def get_model(n_ch, patch_height, patch_width):

    inputs = Input(shape=(n_ch,patch_height,patch_width))
    
    LRUA = 0.25

    # make it channels_last
    inputs_t = Lambda(lambda x: tf.transpose(x, (0,2,3,1)))(inputs)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs_t)
        
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	    
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool4)
        
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)
    
    up1 = UpSampling2D(size=(2, 2))(conv4)
    up1 = concatenate([conv3,up1],axis=3)

    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
        
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    
    up2 = UpSampling2D(size=(2, 2))(conv5)
    up2 = concatenate([conv2,up2], axis=3)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
        
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
	    
    up3 = UpSampling2D(size=(2, 2))(conv6)
    up3 = concatenate([conv1,up2], axis=3)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
        
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    
    conv7 = Conv2D(2, (1, 1), activation='relu',padding='same')(conv7)
    
    # due to channels last, reshape
    conv7 = core.Permute((3,1,2))(conv7)
    conv7 = core.Reshape((2,patch_height*patch_width))(conv7)
    conv7 = core.Permute((2,1))(conv7)
    conv8 = core.Activation('softmax')(conv7)       

    model = Model(input=inputs, output=conv8)

    model.compile(optimizer='sgd', loss='binary_crossentropy',metrics=['accuracy'])

    return model


# list of callbacks to include into model.fit
def get_callbacks():
    callbacks = []
    return callbacks