import numpy as np

class QTMConfig:
    def __init__(self, state, tape, head, amplitude):
        self.state = state          # Estado atual da máquina
        self.tape = tape[:]         # Cópia da fita (lista de símbolos)
        self.head = head            # Posição do cabeçote na fita
        self.amplitude = amplitude  # Amplitude complexa da configuração

    def clone(self):
        return QTMConfig(self.state, self.tape, self.head, self.amplitude)

class QTM:
    def __init__(self, input_string):
        self.blank = '_'
        self.states = ['q0', 'mark', 'check', 'accept', 'reject']
        self.init_state = 'q0'
        self.accept_state = 'accept'
        self.reject_state = 'reject'

        # Alfabeto marcado para símbolos já processados
        self.mark_map = {'a': 'A', 'b': 'B'}

        # Fita inicial: input + blanks extras
        self.tape = list(input_string) + [self.blank]*10
        self.length = len(input_string)

        # Inicializa superposição com configuração inicial
        self.superposition = [QTMConfig(self.init_state, self.tape, 0, 1.0 + 0j)]
        self.max_steps = 2 * len(self.tape) + 10

    def is_final(self, config):
        return config.state in [self.accept_state, self.reject_state]

    def normalize(self, superpos):
        total_prob = sum(abs(cfg.amplitude)**2 for cfg in superpos)
        if total_prob > 0:
            norm_factor = np.sqrt(total_prob)
            for cfg in superpos:
                cfg.amplitude /= norm_factor
        return superpos

    def step(self):
        new_superposition = []

        for cfg in self.superposition:
            if self.is_final(cfg):
                new_superposition.append(cfg)
                continue

            state = cfg.state
            tape = cfg.tape
            head = cfg.head
            amp = cfg.amplitude

            symbol = tape[head] if 0 <= head < len(tape) else self.blank

            # --- Estado q0: marcar primeira metade ---
            if state == 'q0':
                # Se posição cabeçote >= metade, vamos para check
                if head >= self.length // 2:
                    # Mover cabeçote para início da segunda metade para checagem
                    new_cfg = cfg.clone()
                    new_cfg.state = 'check'
                    new_cfg.head = self.length // 2
                    new_superposition.append(new_cfg)
                else:
                    # Marcar símbolo da primeira metade se for 'a' ou 'b'
                    if symbol in self.mark_map:
                        marked = self.mark_map[symbol]
                        new_tape = tape[:]
                        new_tape[head] = marked
                        new_cfg = QTMConfig(state='q0', tape=new_tape, head=head+1, amplitude=amp)
                        new_superposition.append(new_cfg)
                    else:
                        # Se símbolo inesperado, rejeita
                        rej_cfg = cfg.clone()
                        rej_cfg.state = self.reject_state
                        new_superposition.append(rej_cfg)

            # --- Estado check: comparar símbolos ---
            elif state == 'check':
                # Se cabeçote alcançou o fim da fita (ou da entrada), aceita
                if head >= self.length:
                    acc_cfg = cfg.clone()
                    acc_cfg.state = self.accept_state
                    new_superposition.append(acc_cfg)
                    continue

                # Símbolo marcado esperado da primeira metade
                marked_expected = tape[head - self.length // 2]
                # Símbolo atual da segunda metade para comparar
                symbol_second_half = tape[head]

                # Aceita se símbolos corresponderem, rejeita se não
                # Aqui devemos comparar o símbolo marcado na primeira metade com o símbolo na segunda metade
                # marcado 'A' corresponde a 'a', 'B' a 'b'
                if marked_expected == 'A' and symbol_second_half == 'a':
                    new_cfg = cfg.clone()
                    new_cfg.head = head + 1
                    new_superposition.append(new_cfg)
                elif marked_expected == 'B' and symbol_second_half == 'b':
                    new_cfg = cfg.clone()
                    new_cfg.head = head + 1
                    new_superposition.append(new_cfg)
                else:
                    rej_cfg = cfg.clone()
                    rej_cfg.state = self.reject_state
                    new_superposition.append(rej_cfg)

        # Normalizar amplitudes
        self.superposition = self.normalize(new_superposition)

    def run(self):
        for step in range(self.max_steps):
            if all(self.is_final(cfg) for cfg in self.superposition):
                break
            self.step()

        # Calcula probabilidade de aceitação
        prob_accept = sum(abs(cfg.amplitude)**2 for cfg in self.superposition if cfg.state == self.accept_state)
        return prob_accept

if __name__ == '__main__':
    inputs = ['abab', 'aabb', 'aaaa', 'aba', 'abcabc', 'abba', 'baba', 'ababab']

    qtm = None
    for word in inputs:
        qtm = QTM(word)
        prob = qtm.run()
        accepted = prob > 0.5
        print(f"Entrada: '{word}' => {'Aceita' if accepted else 'Rejeita'} (Prob. aceitação: {prob:.4f})")
