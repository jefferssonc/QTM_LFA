import numpy as np

# Define uma configuração quântica: (estado, fita, posição da cabeça)
class QTMConfig:
    def __init__(self, state, tape, head, amplitude):
        self.state = state
        self.tape = tape
        self.head = head
        self.amplitude = amplitude

    def clone(self):
        return QTMConfig(self.state, self.tape[:], self.head, self.amplitude)

# Define a QTM simulada
class QuantumTuringMachine:
    def __init__(self, input_string):
        self.input_string = input_string
        self.blank = '_'
        self.init_state = 'q0'
        self.accept_state = 'qa'
        self.reject_state = 'qr'
        self.max_steps = 30

        # Inicializa fita
        tape = list(input_string) + [self.blank] * 10
        self.superposition = [QTMConfig(self.init_state, tape, 0, 1.0+0j)]

    def is_final(self, config):
        return config.state in [self.accept_state, self.reject_state]

    def measure_acceptance(self):
        # Soma das probabilidades (amplitude^2) dos estados que aceitaram
        accept_amp = sum(abs(cfg.amplitude)**2 for cfg in self.superposition if cfg.state == self.accept_state)
        return accept_amp

    def step(self):
        new_superposition = []

        for cfg in self.superposition:
            if self.is_final(cfg):
                new_superposition.append(cfg)
                continue

            symbol = cfg.tape[cfg.head]

            # Transições simplificadas para verificar se a palavra tem o formato ww
            if cfg.state == 'q0':
                # Divide as possibilidades de forma "quântica"
                middle = len(self.input_string) // 2
                if len(self.input_string) % 2 != 0:
                    # Palavra de tamanho ímpar não pode estar em L
                    new_cfg = cfg.clone()
                    new_cfg.state = self.reject_state
                    new_superposition.append(new_cfg)
                    continue

                # Simula superposição para todos os caminhos possíveis de verificação de ww
                w1 = self.input_string[:middle]
                w2 = self.input_string[middle:]

                if w1 == w2:
                    new_cfg = cfg.clone()
                    new_cfg.state = self.accept_state
                    new_cfg.amplitude *= 1 / np.sqrt(2)
                    new_superposition.append(new_cfg)
                else:
                    new_cfg = cfg.clone()
                    new_cfg.state = self.reject_state
                    new_cfg.amplitude *= 1 / np.sqrt(2)
                    new_superposition.append(new_cfg)

            else:
                # Para simplificação, configurações finais não se alteram
                new_superposition.append(cfg)

        # Normaliza amplitudes
        total_amp = sum(abs(cfg.amplitude)**2 for cfg in new_superposition)
        if total_amp > 0:
            for cfg in new_superposition:
                cfg.amplitude /= np.sqrt(total_amp)

        self.superposition = new_superposition

    def run(self):
        for _ in range(self.max_steps):
            if all(self.is_final(cfg) for cfg in self.superposition):
                break
            self.step()

        acceptance_probability = self.measure_acceptance()
        return acceptance_probability > 0.5  # probabilidade maior que 50% para aceitar


# Teste
inputs = [
    "abab",   # deve aceitar
    "aabb",   # rejeitar
    "aaaa",   # aceitar
    "aba",    # rejeitar (ímpar)
    "abba",   # rejeitar
    "ababab"  # rejeitar (não é w + w)

]

for inp in inputs:
    qtm = QuantumTuringMachine(inp)
    result = qtm.run()
    print(f"Entrada: {inp} => {'Aceita' if result else 'Rejeita'}")
