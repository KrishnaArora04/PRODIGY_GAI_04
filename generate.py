import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = (image / 127.5) - 1.0
    return image

def generate_image(model, input_image):
    prediction = model(input_image[tf.newaxis, ...], training=False)
    prediction = (prediction + 1.0) * 127.5
    return prediction[0]

def display_image(image):
    plt.figure(figsize=(6, 6))
    plt.imshow((image / 2 + 0.5).numpy())
    plt.axis('off')
    plt.show()

def main():
    generator = tf.keras.models.load_model('pix2pix_generator.h5')

    input_image_path = 'path/to/test/images/image1.jpg'
    input_image = load_image(input_image_path)
    generated_image = generate_image(generator, input_image)

    display_image(generated_image)

if __name__ == "__main__":
    main()
