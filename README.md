# Lösen von nichtlinearen Gleichungsystemen mit einem Reinforcement Learning Agent

Dieses Repository enthält eine benutzerdefinierte Umgebung zur Optimierung eines nichtlinearen Gleichungssystems (NSE) unter Verwendung von Reinforcement-Learning-Algorithmen, die von Stable Baselines3 bereitgestellt werden.

## Übersicht

Eine genauere Erkärung der Funktionen und Methoden findet sich in `Beschreibung.md`.

## Verwendung

1. Klonen des Repositorys:

```git clone https://github.com/Nicolas2912/OuS_Projekt.git```

2. Installieren der erforderlichen Abhängigkeiten:

``` pip install gym numpy numdifftools scipy matplotlib stable-baselines3 torch ```

3. Ausführung des Skriptes

``` python solving_nse.py ```

## Ergebnisse

Das trainierte Modell (`<name>`) und die Trainingsprotokolle werden im aktuellen Verzeichnis gespeichert. Der Trainingsfortschritt kann mit `TensorBoard` visualisiert werden.

``` tensorboard --logdir ./tmp/ ```
