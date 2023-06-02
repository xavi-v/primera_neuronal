import numpy
import numpy as np
import tensorflow.python.keras.layers
import matplotlib.pyplot

celcius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array((-40, 14, 32, 46, 59, 72, 100), dtype=float)
capa = tensorflow.keras.layers.Dense(units=1,input_shape=[1])
modelo = tensorflow.keras.Sequential(capa)

modelo.compile(optimizer = tensorflow.keras.optimizers.Adam(0.1),
               loss = 'mean_squared_error')
print("modelo cargado...")
print("-" * 10)
print("comenzando entrenamient0")

historial = modelo.fit(celcius, fahrenheit, epochs=1000, verbose=False)
print("modelo entrenado!! ")
print("Pávlov estaría orgulloso, woof")

import matplotlib.pyplot
matplotlib.pyplot.xlabel("# Epoca")
matplotlib.pyplot.ylabel("Magnitud de pérdida")
matplotlib.pyplot.plot(historial.history["loss"])

print("Hagamos una predicción!")
resultado = modelo.predict([100.0])
print("El resultado es " + str(resultado) + " fahrenheit!")