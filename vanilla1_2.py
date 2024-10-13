import tensorflow as tf
import matplotlib.pyplot as plt
import datagen


x_train, y_train = datagen.hydrogen211(800, 2.5)
x_test, y_test = datagen.hydrogen211(200, 5)

in_dim, out_dim = 3, 1
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='tanh'),
    tf.keras.layers.Dense(10, activation='tanh'),
    tf.keras.layers.Dense(10)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError()
)

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