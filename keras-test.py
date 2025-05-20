from tensorflow import keras
import tensorflow as tf

def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

if __name__ == "__main__":
    fashion = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Normalize the images to a range of 0 to 1 before feeding them to the neural network
    # This is done by dividing the pixel values by 255.0
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    #Build a neural network extrating signifcanting features from the images (Layers)
    keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        # 128 neuros where each one has a score to classifie the category of the image
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ])

    # Configure training
    model.compile(optimizr='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    # Train and raiting the model
    modelfit(train_images, train_labels, epochs=10)

    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=2)
    print(f'\nTest accuracy: {test_accuracy}')

    # Make predictions
    odds = keras.Sequential([
        model, keras.layers.Softmax()
    ])

    predictions = odds.predict(test_images)