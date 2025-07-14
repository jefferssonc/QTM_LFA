import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class QTMConfig:
    def __init__(self, state, tape, head, amplitude, step_info=""):
        self.state = state
        self.tape = tape
        self.head = head
        self.amplitude = amplitude
        self.step_info = step_info

    def clone(self):
        return QTMConfig(self.state, self.tape[:], self.head, self.amplitude, self.step_info)

class QuantumTuringMachine:
    def __init__(self, input_string):
        self.input_string = input_string
        self.blank = '_'
        self.init_state = 'q0'
        self.accept_state = 'qa'
        self.reject_state = 'qr'
        self.max_steps = 10
        self.middle = len(input_string) // 2
        self.superposition = [QTMConfig('q0', list(input_string) + [self.blank]*10, 0, 1.0 + 0j)]
        self.history = []

    def is_final(self, config):
        return config.state in [self.accept_state, self.reject_state]

    def record_step(self):
        state_probs = defaultdict(float)
        for cfg in self.superposition:
            state_probs[cfg.state] += abs(cfg.amplitude)**2
        self.history.append(dict(state_probs))

    def step(self):
        new_superposition = []

        for cfg in self.superposition:
            if self.is_final(cfg):
                new_superposition.append(cfg)
                continue

            if cfg.state == 'q0':
                if len(self.input_string) % 2 != 0:
                    reject_cfg = cfg.clone()
                    reject_cfg.state = self.reject_state
                    new_superposition.append(reject_cfg)
                    continue

                # Ir para q1 onde começa verificação caractere a caractere
                next_cfg = cfg.clone()
                next_cfg.state = 'q1'
                next_cfg.head = 0
                next_cfg.step_info = "Start checking"
                new_superposition.append(next_cfg)


            elif cfg.state == 'q1':

                w1 = self.input_string[:self.middle]

                w2 = self.input_string[self.middle:]

                if w1 == w2:
                    acc_amplitude = 1.0 + 0j  # total interferência construtiva
                else:
                    acc_amplitude = 0.0 + 0j  # interferência destrutiva

                final_cfg = cfg.clone()

                final_cfg.state = self.accept_state

                final_cfg.amplitude *= acc_amplitude  # interfere positivamente ou negativamente

                new_superposition.append(final_cfg)


            elif cfg.state == 'q2':
                # Todos os caminhos agora interferem para decidir se aceitam
                final_cfg = cfg.clone()
                final_cfg.state = self.accept_state
                final_cfg.step_info = "Collapse to final"
                new_superposition.append(final_cfg)

        # Normaliza amplitudes
        total_amp = sum(abs(cfg.amplitude)**2 for cfg in new_superposition)
        if total_amp > 0:
            norm_factor = np.sqrt(total_amp)
            for cfg in new_superposition:
                cfg.amplitude /= norm_factor

        self.superposition = new_superposition
        self.record_step()

    def run(self):
        self.record_step()
        for _ in range(self.max_steps):
            if all(self.is_final(cfg) for cfg in self.superposition):
                break
            self.step()
        return self.measure_acceptance()

    def measure_acceptance(self):
        return sum(abs(cfg.amplitude)**2 for cfg in self.superposition if cfg.state == self.accept_state)

# Visualizações

def plot_superposition(superposition, title="Distribuição Final"):
    state_probs = defaultdict(float)
    for cfg in superposition:
        state_probs[cfg.state] += abs(cfg.amplitude)**2

    states = list(state_probs.keys())
    probs = [state_probs[s] for s in states]

    plt.figure(figsize=(6, 4))
    plt.bar(states, probs, color='purple')
    plt.title(title)
    plt.ylabel("Probabilidade")
    plt.xlabel("Estado")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()

def plot_evolution(history, title="Evolução da Superposição"):
    steps = list(range(len(history)))
    states = set()
    for h in history:
        states.update(h.keys())

    for state in sorted(states):
        probs = [h.get(state, 0) for h in history]
        plt.plot(steps, probs, label=state)

    plt.xlabel("Passos")
    plt.ylabel("Probabilidade")
    plt.title(title)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 🧪 Novos testes
inputs = [
    "abab",     # Aceita
    "aabb",     # Rejeita
    "aaaa",     # Aceita
    "aba",      # Rejeita (ímpar)
    "abba",     # Rejeita
    "ababab",   # Rejeita
    "abcabc",   # Aceita
    "abcab",    # Rejeita (ímpar)
    "ccddccdd", # Aceita
    "ccddcdcd", # Rejeita
    "111111",   # Aceita
    "000111",   # Rejeita
]

for inp in inputs:
    print("="*50)
    print(f"Entrada: {inp}")
    qtm = QuantumTuringMachine(inp)
    acc_prob = qtm.run()
    accepted = acc_prob > 0.5
    print(f"=> {'Aceita' if accepted else 'Rejeita'} (Probabilidade de Aceitação: {acc_prob:.3f})")
    plot_evolution(qtm.history, title=f"Evolução - Entrada: {inp}")
    plot_superposition(qtm.superposition, title=f"Distribuição Final - Entrada: {inp}")