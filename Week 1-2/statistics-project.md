# Examen

## Planteamiento

Para controlar la población de ratas de una ciudad, el gobierno instaló rastreadores GPS en cerca de 100 mil individuos, y los volvió a liberar en alcantarillas distribuidas uniformemente por el territorio.
El equipo de investigación busca poder predecir patrones de contagio de enfermedades transmisibles al humano, y buscar las zonas con mayor número de contagios.

Para una bateria particular, se diseñó la siguiente simulación: 

## Definiciones
1. Se considera que dos ratas estuvieron en contacto si pasaron al menos 30 minutos a una distancia menor de 10 metros. 
2. Data una rata enferma tiene una probabilidad de morir de 0.05. En caso de que esto ocurra, la muerte ocurre entre los 7 y los 25 días del contagio (con una confianza del 95%). 
3. En caso de que no muera, se recupera entre los 10 y 35 días de ser contagiada (con una confianza del 95%). 
4. Los individuos que mueren o se curan dejan de contagiar. 

## Simulación 
1. Se comienza la simulación con 100 individuos contagiados distribuidos uniformemente entre la población (que se observa en los eventos disponibles). 
2. Cada día, en cada contacto de una rata sana con una contagiada, hay una probabilidad de 0.00005 de que la sana sea contagiada. 
3. Se calculan los nuevos contagiados de forma diaria. 
4. El ciclo se repite 30 veces (para poder predecir los contagios en un mes). 

## Resultado 
1. Un histograma en dos dimensiones (latitud y longitud), donde cada rectángulo tiene la cantidad de eventos de geolocalización emitidos por ratas enfermas. 
2. Una línea de tiempo donde en cada día pueda leerse la cantidad de individuos contagiados.
3. Un modelo predictivo que pueda predecir las ubicaciones más probables de contagios entre ratas. 


## Para considerar
- ¿Qué estructura de datos es la más adecuada para representar el problema? 
- ¿Cómo sería la forma más conveniente de dividir el proceso en partes más chicas? ¿Qué parte es más pesada en términos de procesamiento? 
- Haciendo algunos cambios en el algoritmo, y evaluando su impacto, ¿se puede simplificar el proceso -tal vez para hacerlo más rápido o más simple de desarrollar? ¿Cuáles serían las concesiones razonables? 
- ¿Cuáles son los casos bordes? 
- ¿Qué posibles problemas de perfomance podría haber?
