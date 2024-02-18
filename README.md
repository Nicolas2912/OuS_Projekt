# Lösen von nichtlinearen Gleichungsystemen mit einem Reinforcement Learning Agent

## Funktionsweise eines Reinforcement Learning Agents

Die Funktionsweise eines Reinforcement Learning (RL) Agents kann in mehreren Schritten beschrieben werden:

### Beobachtung (Observation)

Der Agent erhält Informationen über die Umgebung, in der er sich befindet. Diese Informationen können Zustände (states) genannt werden und enthalten alle relevanten Daten, die der Agent benötigt, um Entscheidungen zu treffen und Aktionen auszuführen.

### Aktion (Action)

Basierend auf der Beobachtung wählt der Agent eine Aktion aus einer vordefinierten Menge von möglichen Aktionen aus. Die Wahl der Aktion kann deterministisch oder stochastisch sein und hängt von der Strategie des Agents ab.

### Umgebung (Environment)

Der Agent führt die ausgewählte Aktion in der Umgebung aus, und die Umgebung reagiert entsprechend. Die Reaktion der Umgebung kann eine Änderung des Zustands und eine Rückmeldung (Feedback) in Form von Belohnungen oder Bestrafungen umfassen.

### Belohnung (Reward)

Basierend auf der Reaktion der Umgebung erhält der Agent eine Belohnung. Die Belohnung ist ein numerisches Signal, das den Erfolg oder Misserfolg des Agents bei der Ausführung der ausgewählten Aktion in der aktuellen Situation quantifiziert. Das Ziel des Agents ist es, die Gesamtbelohnung im Laufe der Zeit zu maximieren.

### Lernen (Learning)

Der Agent verwendet die Beobachtungen, Aktionen und Belohnungen, um seine Strategie oder Richtlinie (Policy) zu aktualisieren. Dies geschieht durch das Lernen aus Erfahrung, indem der Agent versucht, seine Aktionen basierend auf vergangenen Erfahrungen zu verbessern. Dieser Prozess kann durch verschiedene RL-Algorithmen wie Q-Learning, Policy Gradienten-Methoden oder Deep Reinforcement Learning-Algorithmen wie Deep Q-Networks (DQN) oder Proximal Policy Optimization (PPO) erfolgen.

### Exploration vs. Exploitation

Während des Lernprozesses muss der Agent eine Balance zwischen Exploration (Erkunden neuer Aktionen und Zustände) und Exploitation (Ausführen bekannter Aktionen, die zu guten Belohnungen führen) finden. Dies ist wichtig, um sicherzustellen, dass der Agent sowohl neue Strategien lernt als auch bereits gelerntes Wissen effektiv einsetzt.

Der Prozess der Beobachtung, Aktion, Umgebung, Belohnung und des Lernens wird iterativ durchlaufen, während der Agent mit der Umgebung interagiert und seine Strategie verbessert, um die Gesamtbelohnung zu maximieren. Dieser Prozess des iterativen Lernens wird als "Trial and Error" bezeichnet und ist charakteristisch für das Reinforcement Learning.


## Nichtlineare Gleichungssysteme (NGS)

- **Definition**: Ein System von Gleichungen, bei dem mindestens eine Gleichung nichtlinear ist.
- **Formulierung**: Können algebraisch, differential oder eine Kombination aus beiden sein.
- **Beispiel**: \( f(x) = 0 \), \( g(x, y) = 0 \), usw.
- **Anwendungen**: Finden von Nullstellen, Optimierung, physikalische und ingenieurwissenschaftliche Probleme.

## Schwierigkeiten bei der Lösung nichtlinearer Gleichungssysteme:
- **Mangel an geschlossenen Lösungen**: Nichtlineare Gleichungssysteme haben oft keine geschlossenen Lösungen.
- **Komplexität der Lösungsräume**: Lösungen können in komplexen, multidimensionalen Räumen liegen.
- **Empfindlichkeit gegenüber Anfangsbedingungen**: Chaotisches Verhalten und Abhängigkeit von Anfangsbedingungen.
- **Numerische Instabilität**: Mögliche Instabilität bei der Verwendung von numerischen Verfahren.

## Ansätze zur Lösung nichtlinearer Gleichungssysteme:
1. **Iterative Verfahren**:
   - **Newton-Verfahren**
   - **Broyden-Verfahren**
   - **Fixpunktiteration**
2. **Optimierungsverfahren**:
   - **Gradientenabstieg**
   - **Genetische Algorithmen**
3. **Globale Optimierungsverfahren**:
   - **Simulated Annealing**
   - **Particle Swarm Optimization (PSO)**

## Wichtige Fragestellungen bzgl. des Lösens von NGS mit einem RL-Agent

1. **Modellierung der Umgebung**: Wie kann die Umgebung des Agents so modelliert werden, dass sie die Nichtlinearität des Gleichungssystems angemessen widerspiegelt?
2. **Aktion und Zustand**: Wie ist eine Aktion und wie ist ein Zustand im Kontext nicht NGS definiert?
3. **Belohnungsfunktion**: Wie sieht eine angemessene Belohnungsfunktion aus, damit das Problem möglichst effizient gelöst werden kann?
4. **RL-Alorithmus**: Welcher RL-Algorithmus bietet sich besonders für NGS an? Je nach Problem anderer Algorithmus besser?
5. **Aktionsraum**: Kontinuierlich oder diskret?
