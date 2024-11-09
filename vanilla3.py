import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datagen

class Vanilla3(tf.keras.Model):
    def __init__(self, in_dim:int, out_dim:int):
        super().__init__()
        self.in_dim:int = in_dim
        self.out_dim:int = out_dim
        self.dense1 = tf.keras.layers.Dense(in_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(out_dim, activation='softmax')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return x


model = Vanilla3(10, 10)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy()
)

x_train, y_train = datagen.problem0927_2(10, 800, 10)
x_test, y_test = datagen.problem0927_2(10, 200, 10)
train_loss = []
test_loss = []
epochs = 100
print(x_train.shape)
print(y_train.shape)

for epoch in range(epochs):
    model.fit(x_train, y_train, epochs=1, verbose=0)
    train_loss.append(model.evaluate(x_train, y_train, verbose=0))
    test_loss.append(model.evaluate(x_test, y_test, verbose=0))
    print(f"Epoch {epoch + 1}/{epochs} - Training Loss: {train_loss[-1]:.4f} - Testing Loss: {test_loss[-1]:.4f}")

print(x_test)
print(model.call(x_test))

print()

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(test_loss, label='Testing Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

