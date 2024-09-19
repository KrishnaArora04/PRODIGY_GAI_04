import tensorflow as tf
from data_loader import create_dataset
from models import build_generator, build_discriminator

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(tf.ones_like(disc_real_output), disc_real_output))
    generated_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(tf.zeros_like(disc_generated_output), disc_generated_output))
    return real_loss + generated_loss

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(tf.ones_like(disc_generated_output), disc_generated_output))
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_loss = gan_loss + 100 * l1_loss
    return total_loss

@tf.function
def train_step(generator, discriminator, input_image, target, generator_optimizer, discriminator_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def main():
    train_image_paths = ['path/to/train/images/image1.jpg', 'path/to/train/images/image2.jpg']
    train_mask_paths = ['path/to/train/masks/mask1.jpg', 'path/to/train/masks/mask2.jpg']
    train_dataset = create_dataset(train_image_paths, train_mask_paths)

    generator = build_generator()
    discriminator = build_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    epochs = 10
    for epoch in range(epochs):
        for image, mask in train_dataset:
            train_step(generator, discriminator, image, mask, generator_optimizer, discriminator_optimizer)
        print(f"Epoch {epoch+1} completed.")

    generator.save('pix2pix_generator.h5')
    discriminator.save('pix2pix_discriminator.h5')

if __name__ == "__main__":
    main()
