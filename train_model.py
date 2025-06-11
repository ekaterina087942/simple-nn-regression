import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def generate_data(num_samples=100, noise_std=0.1, seed=42):
    """
    Генерация синтетических данных по формуле Y = 3*X + шум.

    Параметры:
        num_samples (int): количество точек.
        noise_std (float): стандартное отклонение шума.
        seed (int): сид для генератора случайных чисел.

    Возвращает:
        X (np.ndarray): входные данные, shape=(num_samples, 1).
        Y (np.ndarray): целевые значения, shape=(num_samples, 1).
    """
    np.random.seed(seed)
    X = np.linspace(-1, 1, num_samples)
    noise = np.random.normal(0, noise_std, size=X.shape)
    Y = 3 * X + noise

    return X.reshape(-1, 1), Y.reshape(-1, 1)

def build_model():
    """
    Создает простую нейросеть с одним скрытым слоем из 10 нейронов.

    Возвращает:
        model (tf.keras.Model): скомпилированная модель.
    """
    model = Sequential([
        Dense(10, activation='relu', input_shape=(1,)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_loss(history):
    """
    Строит график изменения функции потерь за эпохи обучения.

    Параметры:
        history (tf.keras.callbacks.History): история обучения модели.
    """
    plt.plot(history.history['loss'])
    plt.title('График потерь (loss) за эпохи обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.show()

def main():
    # Генерация данных
    X, Y = generate_data()

    # Построение модели
    model = build_model()

    # Обучение модели
    history = model.fit(X, Y, epochs=100, verbose=1)

    # Визуализация результатов
    plot_loss(history)

if __name__ == '__main__':
    main()
