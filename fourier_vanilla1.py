import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datagen

class FourierVanilla1(tf.keras.Model):
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        self.in_dim:int = in_dim
        self.out_dim:int = out_dim
        fourier_rows = []
        for k in range(in_dim):
            fourier_rows.append(tf.cos(2 * np.pi * k / in_dim * tf.range(0, in_dim, 1, dtype=tf.dtypes.float32)))
            plt.plot(fourier_rows[k])
            plt.show()
        self.fourier = tf.stack(fourier_rows)
        self.fourier = tf.transpose(self.fourier)
        self.dense1 = tf.keras.layers.Dense(in_dim, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(in_dim, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(in_dim)
        self.dense4 = tf.keras.layers.Dense(in_dim, activation='tanh')
        self.dense5 = tf.keras.layers.Dense(2 * in_dim, activation='tanh')
        self.dense6 = tf.keras.layers.Dense(2 * in_dim, activation='tanh')
        self.dense7 = tf.keras.layers.Dense(out_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x_d = self.dense4(x)
        x_f = tf.linalg.matmul(x, self.fourier)
        x = tf.concat([x_d, x_f], axis=-1)
        x = self.dense5(x)
        x = self.dense6(x)
        x = self.dense7(x)
        return x


model = FourierVanilla1(100, 100)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError()
)

x_train, y_train = datagen.problem0927(100, 800, 50)
x_test, y_test = datagen.problem0927(100, 200, 100)
train_loss = []
test_loss = []
epochs = 50

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

