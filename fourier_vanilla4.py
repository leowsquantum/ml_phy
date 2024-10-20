import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datagen

class FourierVanilla4(tf.keras.Model):
    def __init__(self, in_dim:int):
        super().__init__()
        self.in_dim:int = in_dim
        fourier_rows = []
        for k in range(in_dim):
            fourier_rows.append(tf.cos(2 * np.pi * k / in_dim * tf.range(0, in_dim, 1, dtype=tf.dtypes.float32)))
        self.fourier = tf.stack(fourier_rows)
        self.fourier = tf.transpose(self.fourier)
        self.dense1 = tf.keras.layers.Dense(2 * in_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x_f = tf.linalg.matmul(x, self.fourier)
        x = tf.concat([x, x_f], axis=-1)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


model = FourierVanilla4(10)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError()
)

x_train, y_train = datagen.problem0927_2(10, 800, 5)
x_test, y_test = datagen.problem0927_2(10, 200, 10)
train_loss = []
test_loss = []
epochs = 50
print(x_train.shape)
print(y_train.shape)

for epoch in range(epochs):
    model.fit(x_train, y_train, epochs=1, verbose=0)
    train_loss.append(model.evaluate(x_train, y_train, verbose=0))
    test_loss.append(model.evaluate(x_test, y_test, verbose=0))
    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss[-1]:.4f} - Testing Loss: {test_loss[-1]:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

