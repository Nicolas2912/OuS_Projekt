# Erklaerung

- In der Funktion `nse` werden die nichtlinearen Gleichungen als Python-Funktionen definiert und in einem numpy Array gespeichert.
- Die Klasse `CustomEnv(gym.Env)` bildet den Hauptteil, die die Umgebung des nichtlinearen Gleichungssystems abbildet. Der Agent agiert in dieser Umgebung.
  - In der Initialisierung der Klasse wird der Bereich der Aktionen, sowie der der Beobachtungen festgelegt (`action_space` & `observation_space`).
  - Optional kann eingestellt werden, ob vorher die Gleichungen geplottet werden sollen oder nicht (nur für 2D) oder ob die Gleichungen im System als diskrete Gleichungen betrachtet werden sollen (kein merklicher Unterschied zwischen diskrete Funktionen oder kontinuierliche Funktionen erkennbar) 
  - **Aktionen** definiert als Punkte im Raum.
  - **Belohung** definiert als die negative Summe der Distanzen zu allen Gleichungen im Gleichungssystem.
  - Die Methode `CustomEnv.step(action)` führt eine Aktion in der Umgebung aus und gibt den neuen Zustand, die Belohnung und ob das Problem gelöst ist, zurück.
  - Die Methode `CustomEnv.reset(seed=None)` setzt den Zustand der Umgebung zurück.
  - Die Methoden `CustomEnv.get_distance(point)` bzw. `CustomEnv.get_distance_discrete(point)` berechnen jeweils die Distanzen zu den Gleichungen im System. Die Distanzen werden in einem Array gespeichert und dieses wird zurückgegeben.
 
  In der `main` wird die Umgebung und das Model erstellt. Der Parameter `log` gibt an, ob die Ergebnisse des Training/Lernens gespeichert werden sollen, oder nicht. Die Umgebung speichert die besten als auch die guten Punkte/Aktionen ("gut":
  die Summe der Distanzen liegt unter einem festgelegten Schwellenwert), die im Anschluss dann mit den Funktionen aus dem System geplottet werden. Anschließend wird das Model gespeichert.

