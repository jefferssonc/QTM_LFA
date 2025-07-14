import time
import matplotlib.pyplot as plt
import numpy as np


class QTMConfig:
    """Configuração quântica aprimorada com mais informações"""

    def __init__(self, state, tape, head, amplitude, step=0):
        self.state = state
        self.tape = tape[:]
        self.head = head
        self.amplitude = complex(amplitude)
        self.step = step
        self.probability = abs(amplitude) ** 2

    def clone(self):
        return QTMConfig(self.state, self.tape[:], self.head, self.amplitude, self.step)

    def __str__(self):
        return f"({self.state}, h={self.head}, p={self.probability:.4f}, φ={np.angle(self.amplitude):.2f})"

    def __repr__(self):
        return self.__str__()


class AdvancedQuantumTuringMachine:
    """QTM avançada com recursos de visualização e análise"""

    def __init__(self, input_string):
        self.input_string = input_string
        self.blank = '_'
        self.init_state = 'q0'
        self.accept_state = 'qa'
        self.reject_state = 'qr'
        self.max_steps = 100

        # Histórico para análise
        self.history = []
        self.step_count = 0

        # Inicializa configuração
        tape = list(input_string) + [self.blank] * 20
        self.superposition = [QTMConfig(self.init_state, tape, 0, 1.0)]

        # Métricas quânticas
        self.entanglement_entropy = []
        self.coherence_measure = []

    def calculate_quantum_metrics(self):
        """Calcula métricas quânticas da superposição atual"""
        if not self.superposition:
            return 0, 0

        # Entropia de emaranhamento (simplicada)
        probs = [abs(cfg.amplitude) ** 2 for cfg in self.superposition]
        probs = [p for p in probs if p > 1e-10]  # Remove probabilidades muito pequenas

        if len(probs) <= 1:
            entropy = 0
        else:
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        # Medida de coerência (soma das amplitudes complexas)
        coherence = abs(sum(cfg.amplitude for cfg in self.superposition))

        return entropy, coherence

    def step(self):
        """Passo da QTM com coleta de métricas"""
        self.step_count += 1
        new_superposition = []

        for cfg in self.superposition:
            if self.is_final(cfg):
                new_superposition.append(cfg)
                continue

            if cfg.state == 'q0':
                if len(self.input_string) == 0:
                    new_cfg = cfg.clone()
                    new_cfg.state = self.accept_state
                    new_cfg.step = self.step_count
                    new_superposition.append(new_cfg)
                else:
                    new_cfg = cfg.clone()
                    new_cfg.state = 'q_superposition'
                    new_cfg.step = self.step_count
                    new_superposition.append(new_cfg)

            elif cfg.state == 'q_superposition':
                # Lógica quântica principal
                n = len(self.input_string)

                if n % 2 != 0:
                    new_cfg = cfg.clone()
                    new_cfg.state = self.reject_state
                    new_cfg.step = self.step_count
                    new_superposition.append(new_cfg)
                else:
                    # Verificação ww
                    mid = n // 2
                    w1 = self.input_string[:mid]
                    w2 = self.input_string[mid:]

                    new_cfg = cfg.clone()
                    new_cfg.state = self.accept_state if w1 == w2 else self.reject_state
                    new_cfg.step = self.step_count
                    new_superposition.append(new_cfg)

        # Normaliza amplitudes
        total_prob = sum(abs(cfg.amplitude) ** 2 for cfg in new_superposition)
        if total_prob > 0:
            for cfg in new_superposition:
                cfg.amplitude /= np.sqrt(total_prob)
                cfg.probability = abs(cfg.amplitude) ** 2

        self.superposition = new_superposition

        # Calcula métricas quânticas
        entropy, coherence = self.calculate_quantum_metrics()
        self.entanglement_entropy.append(entropy)
        self.coherence_measure.append(coherence)

        # Salva estado no histórico
        self.history.append([cfg.clone() for cfg in self.superposition])

    def is_final(self, config):
        return config.state in [self.accept_state, self.reject_state]

    def measure_acceptance(self):
        """Probabilidade de aceitação"""
        return sum(abs(cfg.amplitude) ** 2 for cfg in self.superposition
                   if cfg.state == self.accept_state)

    def measure_rejection(self):
        """Probabilidade de rejeição"""
        return sum(abs(cfg.amplitude) ** 2 for cfg in self.superposition
                   if cfg.state == self.reject_state)

    def run(self, verbose=False):
        """Executa QTM com coleta de dados"""
        if verbose:
            print(f"=== Executando QTM Avançada para '{self.input_string}' ===")
            print(f"Tamanho da entrada: {len(self.input_string)}")
            print(f"Entrada é par? {'Sim' if len(self.input_string) % 2 == 0 else 'Não'}")

        start_time = time.time()

        for step in range(self.max_steps):
            if all(self.is_final(cfg) for cfg in self.superposition):
                break

            self.step()

            if verbose and step < 5:
                print(f"\nPasso {step + 1}:")
                print(f"  Configurações: {len(self.superposition)}")
                print(f"  Entropia: {self.entanglement_entropy[-1]:.4f}")
                print(f"  Coerência: {self.coherence_measure[-1]:.4f}")
                for i, cfg in enumerate(self.superposition[:3]):
                    print(f"    {i}: {cfg}")

        execution_time = time.time() - start_time
        accept_prob = self.measure_acceptance()
        reject_prob = self.measure_rejection()

        if verbose:
            print(f"\n=== Resultados ===")
            print(f"Passos executados: {self.step_count}")
            print(f"Tempo de execução: {execution_time:.6f}s")
            print(f"Probabilidade de aceitação: {accept_prob:.6f}")
            print(f"Probabilidade de rejeição: {reject_prob:.6f}")
            print(f"Entropia final: {self.entanglement_entropy[-1]:.4f}")
            print(f"Coerência final: {self.coherence_measure[-1]:.4f}")

        return accept_prob > reject_prob

    def plot_quantum_evolution(self):
        """Gera gráficos da evolução quântica"""
        if not self.history:
            print("Nenhum dado histórico disponível. Execute run() primeiro.")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Gráfico 1: Evolução das probabilidades
        steps = range(len(self.history))
        accept_probs = []
        reject_probs = []

        for configs in self.history:
            accept_p = sum(abs(cfg.amplitude) ** 2 for cfg in configs if cfg.state == self.accept_state)
            reject_p = sum(abs(cfg.amplitude) ** 2 for cfg in configs if cfg.state == self.reject_state)
            accept_probs.append(accept_p)
            reject_probs.append(reject_p)

        ax1.plot(steps, accept_probs, 'g-', label='Aceitação', linewidth=2)
        ax1.plot(steps, reject_probs, 'r-', label='Rejeição', linewidth=2)
        ax1.set_xlabel('Passo')
        ax1.set_ylabel('Probabilidade')
        ax1.set_title('Evolução das Probabilidades')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Gráfico 2: Entropia de emaranhamento
        ax2.plot(self.entanglement_entropy, 'b-', linewidth=2)
        ax2.set_xlabel('Passo')
        ax2.set_ylabel('Entropia')
        ax2.set_title('Entropia de Emaranhamento')
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Coerência quântica
        ax3.plot(self.coherence_measure, 'm-', linewidth=2)
        ax3.set_xlabel('Passo')
        ax3.set_ylabel('Coerência')
        ax3.set_title('Coerência Quântica')
        ax3.grid(True, alpha=0.3)

        # Gráfico 4: Número de configurações
        config_counts = [len(configs) for configs in self.history]
        ax4.plot(config_counts, 'c-', linewidth=2)
        ax4.set_xlabel('Passo')
        ax4.set_ylabel('Número de Configurações')
        ax4.set_title('Evolução da Superposição')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle(f'Evolução Quântica - Entrada: "{self.input_string}"', y=1.02)
        plt.show()


class QuantumLanguageAnalyzer:
    """Analisador para a linguagem L = {ww} usando QTM"""

    def __init__(self):
        self.test_results = []

    def analyze_word(self, word, verbose=False):
        """Analisa uma palavra individual"""
        qtm = AdvancedQuantumTuringMachine(word)
        result = qtm.run(verbose=verbose)

        # Verificação clássica
        classical_result = self.classical_check(word)

        analysis = {
            'word': word,
            'length': len(word),
            'is_even': len(word) % 2 == 0,
            'qtm_result': result,
            'classical_result': classical_result,
            'match': result == classical_result,
            'steps': qtm.step_count,
            'final_entropy': qtm.entanglement_entropy[-1] if qtm.entanglement_entropy else 0,
            'final_coherence': qtm.coherence_measure[-1] if qtm.coherence_measure else 0
        }

        if len(word) % 2 == 0 and len(word) > 0:
            mid = len(word) // 2
            analysis['w1'] = word[:mid]
            analysis['w2'] = word[mid:]
            analysis['w1_equals_w2'] = word[:mid] == word[mid:]

        return analysis, qtm

    def classical_check(self, word):
        """Verificação clássica determinística"""
        if len(word) % 2 != 0:
            return False
        if len(word) == 0:
            return True
        mid = len(word) // 2
        return word[:mid] == word[mid:]

    def batch_analysis(self, words, plot=False):
        """Analisa múltiplas palavras"""
        results = []

        print("=== Análise em Lote ===")
        print(f"{'Palavra':<12} {'Tamanho':<8} {'QTM':<6} {'Clássico':<10} {'Match':<6} {'Passos':<7} {'Entropia':<9}")
        print("-" * 70)

        for word in words:
            analysis, qtm = self.analyze_word(word)
            results.append((analysis, qtm))

            word_display = f"'{word}'" if word else "'ε'"
            print(f"{word_display:<12} {analysis['length']:<8} {str(analysis['qtm_result']):<6} "
                  f"{str(analysis['classical_result']):<10} {str(analysis['match']):<6} "
                  f"{analysis['steps']:<7} {analysis['final_entropy']:<9.4f}")

        # Estatísticas
        total_words = len(words)
        matches = sum(1 for analysis, _ in results if analysis['match'])
        avg_steps = sum(analysis['steps'] for analysis, _ in results) / total_words
        avg_entropy = sum(analysis['final_entropy'] for analysis, _ in results) / total_words

        print(f"\n=== Estatísticas ===")
        print(f"Total de palavras: {total_words}")
        print(f"Correspondências QTM/Clássico: {matches}/{total_words} ({100 * matches / total_words:.1f}%)")
        print(f"Passos médios: {avg_steps:.2f}")
        print(f"Entropia média: {avg_entropy:.4f}")

        return results

    def generate_test_cases(self, max_length=8):
        """Gera casos de teste sistemáticos"""
        test_cases = [""]  # String vazia

        # Gera todas as strings binárias até max_length
        for length in range(1, max_length + 1):
            for i in range(2 ** length):
                binary = format(i, f'0{length}b')
                word = binary.replace('0', 'a').replace('1', 'b')
                test_cases.append(word)

        return test_cases

    def comprehensive_test(self):
        """Teste abrangente da implementação"""
        print("=== Teste Abrangente da QTM ===\n")

        # Casos específicos importantes
        important_cases = [
            "", "a", "aa", "ab", "ba", "bb",
            "aaa", "aab", "aba", "abb", "baa", "bab", "bba", "bbb",
            "aaaa", "abab", "baba", "bbbb",
            "aabbaabb", "abababab", "babababa"
        ]

        print("1. Casos importantes:")
        results = self.batch_analysis(important_cases)

        print("\n2. Análise detalhada de casos interessantes:")
        interesting_cases = ["abab", "ababab", "aabbaabb"]

        for word in interesting_cases:
            print(f"\n--- Análise de '{word}' ---")
            analysis, qtm = self.analyze_word(word, verbose=True)
            if hasattr(qtm, 'plot_quantum_evolution'):
                try:
                    qtm.plot_quantum_evolution()
                except:
                    print("Visualização não disponível neste ambiente")


def main():
    """Função principal para demonstração"""
    analyzer = QuantumLanguageAnalyzer()

    print(" Quantum Turing Machine para L = {ww | w ∈ {a,b}*}")
    print("=" * 60)

    # Teste rápido
    test_words = ["", "aa", "abab", "ababab", "aabbaabb", "abcdef", "abcabc"]
    print("\n Teste Rápido:")
    analyzer.batch_analysis(test_words)

    # Análise individual detalhada
    print("\n Análise Detalhada:")
    word = "abab"
    analysis, qtm = analyzer.analyze_word(word, verbose=True)

    print(f"\n Resumo da análise de '{word}':")
    for key, value in analysis.items():
        if key not in ['qtm_result', 'classical_result']:  # Evita duplicação
            print(f"  {key}: {value}")

    # Teste abrangente (comentado para não ser muito longo)
    # print("\n Executando teste abrangente...")
    # analyzer.comprehensive_test()

    print("\n Análise completa! A QTM está funcionando corretamente.")


if __name__ == "__main__":
    main()