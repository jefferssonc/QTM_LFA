import numpy as np

class UnitaryTransition:
    def __init__(self, current_state, read_symbol, next_state, write_symbol, move_dir, amplitude):
        self.current_state = current_state
        self.read_symbol = read_symbol
        self.next_state = next_state
        self.write_symbol = write_symbol
        self.move_dir = move_dir  # 'L', 'R', 'N'
        self.amplitude = amplitude

class ModularQuantumTuringMachine:
    def __init__(self, input_string, transitions):
        self.blank = '_'
        self.tape = list(input_string) + [self.blank]*20
        self.head = 0
        self.state = 'q0'
        self.transitions = transitions
        self.max_steps = 500
        self.superposition = [(self.state, self.head, self.tape[:], 1+0j)]
        self.accept_state = 'qa'
        self.reject_state = 'qr'
        self.input_length = len(input_string)
        self.half_length = self.input_length // 2

    def step(self):
        new_superposition = []

        for (state, head, tape, amp) in self.superposition:
            if state in [self.accept_state, self.reject_state]:
                new_superposition.append((state, head, tape, amp))
                continue

            if head < 0 or head >= len(tape):
                new_superposition.append((self.reject_state, head, tape, amp))
                continue

            symbol = tape[head]
            valid_transitions = [t for t in self.transitions if t.current_state == state and t.read_symbol == symbol]

            if not valid_transitions:
                new_superposition.append((self.reject_state, head, tape, amp))
                continue

            for t in valid_transitions:
                new_tape = tape[:]
                new_tape[head] = t.write_symbol

                new_head = head
                if t.move_dir == 'R':
                    new_head += 1
                elif t.move_dir == 'L':
                    new_head -= 1

                new_superposition.append((
                    t.next_state,
                    new_head,
                    new_tape,
                    amp * t.amplitude
                ))

        total_prob = sum(abs(amp)**2 for (_, _, _, amp) in new_superposition)
        if total_prob > 0:
            norm = np.sqrt(total_prob)
            new_superposition = [(s, h, tape, amp/norm) for (s, h, tape, amp) in new_superposition]

        self.superposition = new_superposition

    def run(self):
        if self.input_length == 0:
            return 1.0

        for _ in range(self.max_steps):
            if all(state in [self.accept_state, self.reject_state] for (state, _, _, _) in self.superposition):
                break
            self.step()

        accept_prob = sum(abs(amp)**2 for (state, _, _, amp) in self.superposition if state == self.accept_state)
        return accept_prob

def create_transitions(input_length):
    half_length = input_length // 2
    transitions = []

    for i in range(half_length):
        state = f'q0_{i}'
        next_state = f'q0_{i+1}' if i < half_length-1 else 'q1'

        transitions.append(UnitaryTransition(state, 'a', next_state, 'X', 'R', 1.0))
        transitions.append(UnitaryTransition(state, 'b', next_state, 'Y', 'R', 1.0))

    for i in range(half_length):
        state = f'q1_{i}'
        next_state_cmp = f'q_cmp_{i}'

        transitions.append(UnitaryTransition(state, 'X', next_state_cmp, 'X', 'N', 1.0))
        transitions.append(UnitaryTransition(state, 'Y', next_state_cmp, 'Y', 'N', 1.0))
        transitions.append(UnitaryTransition(state, '_', 'qa', '_', 'N', 1.0))

    return transitions

class SimpleQTM:
    def __init__(self, input_string):
        self.input_string = input_string
        self.accept_state = 'qa'
        self.reject_state = 'qr'
        self.max_steps = 10
        self.superposition = [(input_string, 1+0j)]
        self.length = len(input_string)

    def run(self):
        if self.length == 0:
            return 1.0
        if any(c not in ('a', 'b') for c in self.input_string):
            return 0.0
        if self.length % 2 != 0:
            return 0.0

        mid = self.length // 2
        w1 = self.input_string[:mid]
        w2 = self.input_string[mid:]
        if w1 == w2:
            return 1.0
        else:
            return 0.0

inputs = [
    "abab",
    "aabb",
    "aaaa",
    "aba",
    "abcabc",
    "abba",
    "baba",
    "ababab",
    "",
]

print("Teste QTM simplificada para linguagem L = { ww | w ∈ {a,b}* }")
for w in inputs:
    qtm = SimpleQTM(w)
    p_accept = qtm.run()
    print(f"Entrada: '{w:<6}' => {'Aceita' if p_accept > 0.5 else 'Rejeita'} (Probabilidade: {p_accept:.4f})")
