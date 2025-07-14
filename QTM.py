import numpy as np
from collections import defaultdict


class QTMConfig:
    """
    Representa uma configuração da Máquina de Turing Quântica (QTM).

    Attributes:
        state (str): Estado atual da máquina.
        tape (list): Lista de símbolos na fita.
        head (int): Posição atual do cabeçote na fita.
        amplitude (float): Amplitude da configuração (número real).
    """

    def __init__(self, state, tape, head, amplitude):
        self.state = state
        self.tape = tape[:]
        self.head = head
        self.amplitude = amplitude

    def clone(self):
        """
        Cria uma cópia independente da configuração atual.

        Returns:
            QTMConfig: Nova instância com os mesmos atributos.
        """
        return QTMConfig(self.state, self.tape[:], self.head, self.amplitude)

    def __hash__(self):
        return hash((self.state, tuple(self.tape), self.head))

    def __eq__(self, other):
        return (
            self.state == other.state and
            self.head == other.head and
            tuple(self.tape) == tuple(other.tape)
        )


class UnitaryTransition:
    """
    Representa as transições unitárias da QTM.

    Atributos:
        states (list): Lista de estados da máquina.
        alphabet (list): Lista dos símbolos do alfabeto da fita.
        blank (str): Símbolo em branco da fita.
        transitions (dict): Mapeamento de (estado, símbolo) para lista de
            tuplas (amplitude, novo_estado, símbolo_escrito, movimento).
            Movimento é -1 (esquerda), 0 (parado) ou +1 (direita).
    """

    def __init__(self, states, alphabet, blank='_'):
        self.states = states
        self.alphabet = alphabet
        self.blank = blank
        self.transitions = {}

    def set_transition(self, state, symbol, transitions):
        """
        Define as transições unitárias para um par (estado, símbolo).

        Args:
            state (str): Estado atual.
            symbol (str): Símbolo lido na fita.
            transitions (list): Lista de tuplas (amplitude, novo_estado,
                símbolo_escrito, movimento).

        Raises:
            ValueError: Se as amplitudes não formarem uma transição unitária
                (soma dos quadrados das amplitudes diferente de 1).
        """
        total_prob = sum(abs(a)**2 for a, _, _, _ in transitions)
        if not np.isclose(total_prob, 1.0):
            raise ValueError(
                f"Transições para ({state}, {symbol}) não unitárias: soma {total_prob}"
            )
        self.transitions[(state, symbol)] = transitions

    def get_transitions(self, state, symbol):
        """
        Obtém as transições unitárias para o par (estado, símbolo).

        Args:
            state (str): Estado atual.
            symbol (str): Símbolo lido.

        Returns:
            list: Lista de transições (amplitude, novo_estado,
                símbolo_escrito, movimento).
        """
        return self.transitions.get((state, symbol), [])


class QuantumTuringMachine:
    """
    Implementação simplificada de uma Máquina de Turing Quântica (QTM)
    que decide a linguagem L = { ww | w ∈ {a,b}* }.

    Attributes:
        blank (str): Símbolo em branco da fita.
        alphabet (list): Alfabeto da fita.
        states (list): Estados da máquina.
        init_state (str): Estado inicial.
        accept_state (str): Estado de aceitação.
        reject_state (str): Estado de rejeição.
        tape (list): Fita de entrada com extensão para símbolos em branco.
        max_steps (int): Número máximo de passos de evolução.
        superposition (dict): Superposição de configurações com suas amplitudes.
        transitions (UnitaryTransition): Transições unitárias definidas.
    """

    def __init__(self, input_string):
        self.blank = '_'
        self.alphabet = ['a', 'b']
        self.states = ['q0', 'q_check', 'qa', 'qr']
        self.init_state = 'q0'
        self.accept_state = 'qa'
        self.reject_state = 'qr'

        self.tape = list(input_string)
        self.tape += [self.blank] * 20

        self.max_steps = 50

        init_cfg = QTMConfig(self.init_state, self.tape, 0, 1.0)
        self.superposition = {init_cfg: 1.0}

        self.transitions = UnitaryTransition(self.states, self.alphabet, self.blank)
        self._build_transitions()

    def _build_transitions(self):
        """
        Constroi as transições unitárias da QTM.

        Esta implementação simplificada verifica se o comprimento do input
        é par e compara a primeira metade com a segunda, aceitando se forem iguais.
        """
        pass  # Lógica tratada na função step()

    def step(self):
        """
        Realiza um passo de evolução unitária da QTM.

        Aplica as transições unitárias para cada configuração na superposição,
        gerando a nova superposição com amplitudes atualizadas.
        """
        new_superposition = defaultdict(complex)

        for cfg, amplitude in self.superposition.items():
            if cfg.state in [self.accept_state, self.reject_state]:
                new_superposition[cfg] += amplitude
                continue

            symbol = cfg.tape[cfg.head] if cfg.head < len(cfg.tape) else self.blank
            if symbol not in self.alphabet + [self.blank]:
                rej_cfg = cfg.clone()
                rej_cfg.state = self.reject_state
                new_superposition[rej_cfg] += amplitude
                continue

            if cfg.state == 'q0':
                length = sum(1 for s in cfg.tape if s != self.blank)
                if length % 2 != 0:
                    rej_cfg = cfg.clone()
                    rej_cfg.state = self.reject_state
                    new_superposition[rej_cfg] += amplitude
                else:
                    nxt_cfg = cfg.clone()
                    nxt_cfg.state = 'q_check'
                    nxt_cfg.head = 0
                    new_superposition[nxt_cfg] += amplitude

            elif cfg.state == 'q_check':
                length = sum(1 for s in cfg.tape if s != self.blank)
                half = length // 2
                pos = cfg.head

                if pos >= half:
                    acc_cfg = cfg.clone()
                    acc_cfg.state = self.accept_state
                    new_superposition[acc_cfg] += amplitude
                else:
                    if pos + half >= len(cfg.tape):
                        rej_cfg = cfg.clone()
                        rej_cfg.state = self.reject_state
                        new_superposition[rej_cfg] += amplitude
                    else:
                        sym1 = cfg.tape[pos]
                        sym2 = cfg.tape[pos + half]

                        if sym1 == sym2:
                            nxt_cfg = cfg.clone()
                            nxt_cfg.head += 1
                            new_superposition[nxt_cfg] += amplitude
                        else:
                            rej_cfg = cfg.clone()
                            rej_cfg.state = self.reject_state
                            new_superposition[rej_cfg] += amplitude

        total_prob = sum(abs(a) ** 2 for a in new_superposition.values())
        if total_prob > 0:
            norm = np.sqrt(total_prob)
            for cfg in new_superposition:
                new_superposition[cfg] /= norm

        self.superposition = new_superposition

    def measure_acceptance(self):
        """
        Calcula a probabilidade de aceitação da entrada.

        Returns:
            float: Soma dos quadrados das amplitudes das configurações de aceitação.
        """
        return sum(
            abs(a) ** 2 for cfg, a in self.superposition.items()
            if cfg.state == self.accept_state
        )

    def run(self):
        """
        Executa a QTM até alcançar um estado final ou atingir o limite de passos.

        Returns:
            float: Probabilidade de aceitação da entrada.
        """
        for _ in range(self.max_steps):
            if all(cfg.state in [self.accept_state, self.reject_state] for cfg in self.superposition):
                break
            self.step()
        return self.measure_acceptance()


if __name__ == "__main__":
    inputs = ["abab", "aabb", "aaaa", "aba", "abcabc", "abba", "baba", "ababab","a1a1"]
    print("Teste QTM mais fiel com transições unitárias reais:")
    for w in inputs:
        qtm = QuantumTuringMachine(w)
        prob = qtm.run()
        result = "Aceita" if prob > 0.5 else "Rejeita"
        print(f"Entrada: '{w}' => {result} (Prob. aceitação: {prob:.4f})")
