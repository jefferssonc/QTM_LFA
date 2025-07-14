import numpy as np
import matplotlib.pyplot as plt


class QTMConfig:
    """
    Representa uma configuração da Máquina de Turing Quântica.

    Attributes:
        state (str): Estado atual da máquina.
        tape (list): Lista de símbolos na fita.
        head (int): Posição do cabeçote na fita.
        amplitude (complex): Amplitude da configuração.
        halt_qubit (int): Indicador de parada (0 = em execução, 1 = parada).
    """

    def __init__(self, state, tape, head, amplitude, halt_qubit=0):
        self.state = state
        self.tape = tape[:]
        self.head = head
        self.amplitude = complex(amplitude)
        self.halt_qubit = halt_qubit

    def clone(self):
        """
        Cria uma cópia da configuração atual.

        Returns:
            QTMConfig: Nova instância com os mesmos valores.
        """
        return QTMConfig(self.state, self.tape[:], self.head, self.amplitude, self.halt_qubit)


class UnitaryTransition:
    """
    Representa uma matriz unitária real para transições locais da QTM.

    Atributos:
        states (list): Lista de estados da máquina.
        symbols (list): Lista dos símbolos do alfabeto.
        dim (int): Dimensão da matriz unitária.
        index_map (dict): Mapeamento de pares (estado, símbolo) para índices.
        U (np.ndarray): Matriz unitária (inicialmente identidade).
    """

    def __init__(self):
        self.states = ['q0', 'q_check', 'qa', 'qr']
        self.symbols = ['a', 'b', '_']
        self.dim = len(self.states) * len(self.symbols)
        self.index_map = {}
        idx = 0
        for s in self.states:
            for sym in self.symbols:
                self.index_map[(s, sym)] = idx
                idx += 1

        self.U = np.identity(self.dim)

    def set_transition(self, from_state, from_sym, to_state, to_sym, amplitude):
        """
        Define a amplitude de transição entre pares estado/símbolo.

        Args:
            from_state (str): Estado de origem.
            from_sym (str): Símbolo de origem.
            to_state (str): Estado de destino.
            to_sym (str): Símbolo de destino.
            amplitude (float): Amplitude associada à transição.
        """
        i = self.index_map[(from_state, from_sym)]
        j = self.index_map[(to_state, to_sym)]
        self.U[i, j] = amplitude

    def validate_unitary(self):
        """
        Verifica se a matriz U é unitária (U * U^T = I).

        Returns:
            bool: True se for unitária, False caso contrário.
        """
        prod = np.dot(self.U, self.U.T)
        return np.allclose(prod, np.identity(self.dim), atol=1e-8)

    def apply(self, state_vec):
        """
        Aplica a matriz unitária a um vetor de estado.

        Args:
            state_vec (np.ndarray): Vetor de estado.

        Returns:
            np.ndarray: Novo vetor após aplicação da matriz.
        """
        return np.dot(state_vec, self.U)


class QuantumTuringMachine:
    """
    Máquina de Turing Quântica simplificada para linguagem L = { ww | w ∈ {a,b}* }.

    Attributes:
        blank (str): Símbolo em branco da fita.
        alphabet (set): Conjunto de símbolos do alfabeto.
        input_string (str): Cadeia de entrada.
        tape (list): Fita da máquina.
        head (int): Posição do cabeçote.
        states (list): Lista de estados.
        init_state (str): Estado inicial.
        accept_state (str): Estado de aceitação.
        reject_state (str): Estado de rejeição.
        max_steps (int): Máximo de passos permitidos.
        superposition (list): Superposição de configurações atuais.
        unitary (UnitaryTransition): Matriz unitária para transições.
        history (list): Histórico das superposições em cada passo.
    """

    def __init__(self, input_string):
        self.blank = '_'
        self.alphabet = {'a', 'b'}
        self.input_string = input_string
        self.tape = list(input_string) + [self.blank] * 20
        self.head = 0
        self.states = ['q0', 'q_check', 'qa', 'qr']
        self.init_state = 'q0'
        self.accept_state = 'qa'
        self.reject_state = 'qr'
        self.max_steps = 100

        self.superposition = [QTMConfig(self.init_state, self.tape, 0, 1.0, halt_qubit=0)]

        self.unitary = UnitaryTransition()

        self.history = []

    def normalize(self):
        """
        Normaliza as amplitudes das configurações para que a soma dos quadrados seja 1.
        """
        norm = np.sqrt(sum(abs(cfg.amplitude) ** 2 for cfg in self.superposition))
        if norm > 0:
            for cfg in self.superposition:
                cfg.amplitude /= norm

    def is_final(self, cfg):
        """
        Verifica se uma configuração está em estado de parada.

        Args:
            cfg (QTMConfig): Configuração atual.

        Returns:
            bool: True se a configuração está em estado halt, False caso contrário.
        """
        return cfg.halt_qubit == 1

    def step(self):
        """
        Executa um passo de evolução da QTM aplicando as regras de transição.
        """
        new_superposition = []

        for cfg in self.superposition:
            if self.is_final(cfg):
                new_superposition.append(cfg.clone())
                continue

            if cfg.tape[cfg.head] not in self.alphabet and cfg.tape[cfg.head] != self.blank:
                new_cfg = cfg.clone()
                new_cfg.state = self.reject_state
                new_cfg.halt_qubit = 1
                new_superposition.append(new_cfg)
                continue

            if cfg.state == 'q0':
                if len(self.input_string) % 2 != 0:
                    new_cfg = cfg.clone()
                    new_cfg.state = self.reject_state
                    new_cfg.halt_qubit = 1
                    new_superposition.append(new_cfg)
                else:
                    new_cfg = cfg.clone()
                    new_cfg.state = 'q_check'
                    new_cfg.halt_qubit = 0
                    new_superposition.append(new_cfg)
                continue

            if cfg.state == 'q_check':
                mid = len(self.input_string) // 2
                w1 = cfg.tape[:mid]
                w2 = cfg.tape[mid : mid * 2]
                if w1 == w2 and all(c in self.alphabet or c == self.blank for c in w1 + w2):
                    new_cfg = cfg.clone()
                    new_cfg.state = self.accept_state
                    new_cfg.halt_qubit = 1
                    new_superposition.append(new_cfg)
                else:
                    new_cfg = cfg.clone()
                    new_cfg.state = self.reject_state
                    new_cfg.halt_qubit = 1
                    new_superposition.append(new_cfg)
                continue

        self.superposition = new_superposition
        self.normalize()
        self.history.append([cfg.clone() for cfg in self.superposition])

    def run(self):
        """
        Executa a QTM até que todas as configurações estejam em estado final ou atinja o limite de passos.

        Returns:
            float: Probabilidade de aceitação da entrada.
        """
        self.history.append([cfg.clone() for cfg in self.superposition])
        for _ in range(self.max_steps):
            if all(self.is_final(cfg) for cfg in self.superposition):
                break
            self.step()
        return self.measure_acceptance()

    def measure_acceptance(self):
        """
        Calcula a probabilidade de aceitação.

        Returns:
            float: Soma dos quadrados das amplitudes das configurações em estado de aceitação.
        """
        return sum(abs(cfg.amplitude) ** 2 for cfg in self.superposition if cfg.state == self.accept_state)

    def measure_rejection(self):
        """
        Calcula a probabilidade de rejeição.

        Returns:
            float: Soma dos quadrados das amplitudes das configurações em estado de rejeição.
        """
        return sum(abs(cfg.amplitude) ** 2 for cfg in self.superposition if cfg.state == self.reject_state)

    def plot_evolution(self):
        """
        Plota a evolução das probabilidades de aceitação, rejeição e o tamanho da superposição ao longo dos passos.
        """
        steps = list(range(len(self.history)))
        accept_probs = []
        reject_probs = []
        num_configs = []

        for configs in self.history:
            accept_p = sum(abs(cfg.amplitude) ** 2 for cfg in configs if cfg.state == self.accept_state)
            reject_p = sum(abs(cfg.amplitude) ** 2 for cfg in configs if cfg.state == self.reject_state)
            accept_probs.append(accept_p)
            reject_probs.append(reject_p)
            num_configs.append(len(configs))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(steps, accept_probs, label='Aceitação', color='green')
        plt.plot(steps, reject_probs, label='Rejeição', color='red')
        plt.xlabel('Passos')
        plt.ylabel('Probabilidade')
        plt.title('Probabilidade de Aceitação e Rejeição')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(steps, num_configs, label='Configurações na Superposição', color='blue')
        plt.xlabel('Passos')
        plt.ylabel('Número de Configurações')
        plt.title('Tamanho da Superposição')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    inputs = [
        'abab', 'aabb', 'aaaa', 'aba', 'abcabc', 'abba', 'baba', 'ababab', 'abababab',
        'a', 'bb', 'abc', 'ab1ab1', 'cc', 'acac', 'aaa', 'bbb', 'aaaa', 'bbbb', 'baba', 'a1a1',
    ]

    print("Teste QTM com evolução unitária local, parada coerente e fita dinâmica")
    for inp in inputs:
        qtm = QuantumTuringMachine(inp)
        prob = qtm.run()
        result = "Aceita" if prob > 0.5 else "Rejeita"
        print(f"Entrada: '{inp}' => {result} (Prob. aceitação: {prob:.4f})")
        qtm.plot_evolution()
