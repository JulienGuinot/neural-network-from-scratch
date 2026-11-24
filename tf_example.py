


model =tf.keras.Sequential([
    tf.keras.Input(shape=(128,128,3)),
    tf.keras.Input(shape=(128, 128, 3)),
    tf.keras.layers.RandomFlip(mode="horizontal"),
    tf.keras.layers.RandomTranslation(0.2,0.2)
    # tf.keras.layers.Rescaling(1./255), # Redundant, already done in preprocess

    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2,2),


    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"), # Standard hidden Dense layer
    tf.keras.layers.Dense(1, activation="sigmoid") # Output for binary classification

])


model.summary()




model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
