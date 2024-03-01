# Lösen von nichtlinearen Gleichungsystemen mit einem Reinforcement Learning Agent

Dieses Repository enthält eine benutzerdefinierte Umgebung zur Optimierung eines nichtlinearen Gleichungssystems (NSE) unter Verwendung von Reinforcement-Learning-Algorithmen, die von Stable Baselines3 bereitgestellt werden.

## Übersicht

Eine genauere Erkärung der Funktionen und Methoden findet sich in `Beschreibung.md`.

## Verwendung

1. Klonen des Repositorys:

```git clone https://github.com/Nicolas2912/OuS_Projekt.git```

2. Installieren der erforderlichen Abhängigkeiten:

``` pip install gymnasium numpy scipy matplotlib stable-baselines3[extra] torch ```

In manchen Shells wie z.B. zsh ist es notwendig Anführungszeichen zu verwenden. 

Z.B.: `pip install stable-baselines3[extra]`

Notiz: Bei der Installation von `stable-baselines3[extra]` kann es dazu kommen, dass `grpcio` nicht installiert werden kann
und die Installation nicht abgeschlossen werden kann. Um diesen Fehler zu beheben, kann folgender Befehl verwendet werden:

``` pip install --only-binary ":all:" grpcio ```

3. Ausführung des Skriptes

``` python solving_nse.py ```

## Ergebnisse

Das trainierte Modell (`<name>`) und die Trainingsprotokolle werden im aktuellen Verzeichnis gespeichert. Der Trainingsfortschritt kann mit `TensorBoard` (sofern installiert) visualisiert werden.

``` tensorboard --logdir ./tmp/ ```
