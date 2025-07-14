import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class QTMConfig:
    def __init__(self, state, head, amplitude, w1, w2):
        self.state = state
        self.head = head
        self.amplitude = amplitude
        self.w1 = w1
        self.w2 = w2

    def clone(self):
        return QTMConfig(self.state, self.head, self.amplitude, self.w1, self.w2)

class QuantumTuringMachine:
    def __init__(self, input_string):
        self.input_string = input_string
        self.accept_state = 'qa'
        self.reject_state = 'qr'
        self.check_state = 'q_check'
        self.max_steps = 20

        if len(input_string) % 2 != 0:
            self.superposition = [QTMConfig(self.reject_state, 0, 1.0 + 0j, '', '')]
        else:
            mid = len(input_string) // 2
            w1 = input_string[:mid]
            w2 = input_string[mid:]
            self.superposition = [
                QTMConfig(self.check_state, 0, 1.0 / np.sqrt(2), w1, w2),
                QTMConfig(self.check_state, 0, -1.0 / np.sqrt(2), w1, w2)
            ]

    def is_final(self, config):
        return config.state in [self.accept_state, self.reject_state]

    def measure_acceptance(self):
        return sum(abs(cfg.amplitude)**2 for cfg in self.superposition if cfg.state == self.accept_state)

    def step(self):
        new_superposition = []

        for cfg in self.superposition:
            if self.is_final(cfg):
                new_superposition.append(cfg)
                continue

            if cfg.state == self.check_state:
                i = cfg.head
                if i >= len(cfg.w1):
                    # Verificação concluída
                    new_cfg = cfg.clone()
                    new_cfg.state = self.accept_state
                    new_superposition.append(new_cfg)
                else:
                    # Comparação caractere a caractere
                    if cfg.w1[i] != cfg.w2[i]:
                        new_cfg = cfg.clone()
                        new_cfg.state = self.reject_state
                        new_superposition.append(new_cfg)
                    else:
                        next_cfg = cfg.clone()
                        next_cfg.head += 1
                        new_superposition.append(next_cfg)

        # Normalização
        total_amp = sum(abs(cfg.amplitude)**2 for cfg in new_superposition)
        if total_amp > 0:
            norm = np.sqrt(total_amp)
            for cfg in new_superposition:
                cfg.amplitude /= norm

        self.superposition = new_superposition

    def run(self):
        for _ in range(self.max_steps):
            if all(self.is_final(cfg) for cfg in self.superposition):
                break
            self.step()
        return self.measure_acceptance()

def plot_superposition(superposition, title="Distribuição de Probabilidades"):
    state_probs = defaultdict(float)
    for cfg in superposition:
        state_probs[cfg.state] += abs(cfg.amplitude)**2

    states = list(state_probs.keys())
    probs = [state_probs[s] for s in states]

    plt.figure(figsize=(6, 4))
    plt.bar(states, probs, color='darkcyan')
    plt.title(title)
    plt.ylabel("Probabilidade")
    plt.xlabel("Estado")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

# Testes
inputs = ["abab", "aabb", "aaaa", "aba", "abba", "ababab"]
for inp in inputs:
    qtm = QuantumTuringMachine(inp)
    acc_prob = qtm.run()
    accepted = acc_prob > 0.5
    print(f"Entrada: {inp} => {'Aceita' if accepted else 'Rejeita'} (Probabilidade de Aceitação: {acc_prob:.3f})")
    plot_superposition(qtm.superposition, title=f"Entrada: {inp}")
