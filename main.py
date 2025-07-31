import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import torch.nn.functional as F

class ExplosiveNeuralNetwork:
    """
    Implementation of curved neural networks exhibiting explosive phase transitions
    based on deformed exponential family distributions using Rényi entropy.
    """
    
    def __init__(self, N: int, gamma_prime: float = -1.0, beta: float = 1.0, 
                 device: str = 'cpu'):
        """
        Initialize the explosive neural network.
        
        Args:
            N: Number of neurons
            gamma_prime: Deformation parameter (negative for explosive behavior)
            beta: Inverse temperature
            device: Computing device ('cpu' or 'cuda')
        """
        self.N = N
        self.gamma_prime = gamma_prime
        self.beta = beta
        self.device = device
        
        # Initialize weights and biases
        self.J = torch.zeros(N, N, device=device)  # Coupling matrix
        self.H = torch.zeros(N, device=device)     # External field
        
        # Pattern storage
        self.patterns = []
        self.M = 0  # Number of stored patterns
        
    def store_patterns(self, patterns: torch.Tensor) -> None:
        """
        Store patterns using Hebbian learning rule.
        
        Args:
            patterns: Tensor of shape (M, N) with M patterns of N neurons
        """
        patterns = patterns.to(self.device)
        self.patterns = patterns
        self.M = patterns.shape[0]
        
        # Hebbian rule: J_ij = sum_a xi^a_i * xi^a_j
        self.J = torch.zeros(self.N, self.N, device=self.device)
        for pattern in patterns:
            self.J += torch.outer(pattern, pattern)
        
        # Zero diagonal (no self-connections)
        self.J.fill_diagonal_(0)
        self.J /= self.N  # Normalize
        
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate the energy function E(x) = -H·x - (1/2N)x·J·x
        
        Args:
            x: Neural state tensor of shape (..., N)
            
        Returns:
            Energy values
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        # Fixed: Use matmul for batch operations
        field_term = torch.matmul(x, self.H)
        interaction_term = 0.5 * torch.sum(x * torch.matmul(x, self.J), dim=-1)
        
        return -(field_term + interaction_term)
    
    def effective_temperature(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculate effective inverse temperature β'(x) = β / (1 - γβE(x))
        
        Args:
            x: Neural state tensor
            
        Returns:
            Effective inverse temperature
        """
        E = self.energy(x)
        denominator = 1 - self.gamma_prime * self.beta * E / self.N
        return self.beta / torch.clamp(denominator, min=1e-8)
    
    def local_field(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """
        Calculate local field h_i = H_i + sum_j J_ij * x_j
        
        Args:
            x: Neural state tensor
            i: Neuron index
            
        Returns:
            Local field for neuron i
        """
        # Fixed: Use matmul for batch operations
        return self.H[i] + torch.matmul(x, self.J[:, i])
    
    def glauber_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single Glauber dynamics update step with curved activation.
        
        Args:
            x: Current state tensor of shape (batch_size, N)
            
        Returns:
            Updated state tensor
        """
        x_new = x.clone()
        batch_size = x.shape[0]
        
        for batch_idx in range(batch_size):
            # Random neuron selection
            i = torch.randint(0, self.N, (1,)).item()
            
            # Calculate effective temperature
            beta_eff = self.effective_temperature(x[batch_idx])
            
            # Calculate local field
            h_i = self.local_field(x[batch_idx], i)
            
            # Energy difference for flipping neuron i
            delta_E = 2 * x[batch_idx, i] * h_i
            
            # Curved activation function
            if self.gamma_prime != 0:
                arg = -self.gamma_prime * beta_eff * delta_E
                if arg > -1/self.gamma_prime:
                    prob = 1 / (1 + torch.pow(1 + self.gamma_prime * (-beta_eff * delta_E), 1/self.gamma_prime))
                else:
                    prob = torch.tensor(1.0)
            else:
                # Standard Glauber dynamics (γ = 0)
                prob = torch.sigmoid(-beta_eff * delta_E)
            
            # Update with probability
            if torch.rand(1) < prob:
                x_new[batch_idx, i] = -x[batch_idx, i]
                
        return x_new
    
    def simulate_dynamics(self, x_init: torch.Tensor, steps: int) -> Tuple[torch.Tensor, List[float]]:
        """
        Simulate Glauber dynamics for specified number of steps.
        
        Args:
            x_init: Initial state tensor
            steps: Number of simulation steps
            
        Returns:
            Tuple of (final_states, energy_history)
        """
        x = x_init.clone()
        energy_history = []
        
        for step in range(steps):
            x = self.glauber_step(x)
            if step % 10 == 0:  # Record every 10 steps
                avg_energy = torch.mean(self.energy(x)).item()
                energy_history.append(avg_energy)
                
        return x, energy_history

# Keep the rest of the classes unchanged
class MeanFieldAnalysis:
    """
    Mean-field analysis for explosive neural networks.
    """
    
    def __init__(self, gamma_prime: float = -1.0):
        self.gamma_prime = gamma_prime
    
    def single_pattern_solution(self, beta_range: np.ndarray, J: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve mean-field equations for single pattern case.
        
        Args:
            beta_range: Array of inverse temperatures
            J: Coupling strength
            
        Returns:
            Tuple of (magnetizations, effective_temperatures)
        """
        magnetizations = []
        beta_effs = []
        
        for beta in beta_range:
            # Iterative solution of m = tanh(β'm), β' = β/(1 + γ'Jm²/2)
            m = 0.1  # Initial guess
            
            for _ in range(1000):  # Fixed-point iteration
                beta_eff = beta / (1 + self.gamma_prime * J * m**2 / 2)
                m_new = np.tanh(beta_eff * J * m)
                
                if abs(m_new - m) < 1e-8:
                    break
                m = m_new
                
            magnetizations.append(m)
            beta_effs.append(beta_eff)
            
        return np.array(magnetizations), np.array(beta_effs)

# Keep all remaining functions exactly the same
def create_binary_patterns(M: int, N: int, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    return torch.randint(0, 2, (M, N)).float() * 2 - 1

def plot_phase_transition(beta_range: np.ndarray, gamma_values: List[float]):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, gamma in enumerate(gamma_values):
        mf_analysis = MeanFieldAnalysis(gamma)
        m_vals, beta_eff_vals = mf_analysis.single_pattern_solution(beta_range)
        
        axes[idx].plot(beta_range, m_vals, 'b-', linewidth=2, label='Forward')
        m_vals_back, _ = mf_analysis.single_pattern_solution(beta_range[::-1])
        axes[idx].plot(beta_range, m_vals_back[::-1], 'r--', linewidth=2, label='Backward')
        
        axes[idx].set_xlabel('β (Inverse Temperature)')
        axes[idx].set_ylabel('Magnetization m')
        axes[idx].set_title(f'Phase Transition (γ\' = {gamma})')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
        
        if gamma < 0:
            axes[idx].set_facecolor('#fff5f5')
            
    plt.tight_layout()
    plt.show()

def demonstrate_memory_retrieval():
    N = 100
    M = 3
    
    patterns = create_binary_patterns(M, N, seed=42)
    gamma_values = [0.0, -0.5, -1.0, -1.5]
    networks = [ExplosiveNeuralNetwork(N, gamma, beta=2.0) for gamma in gamma_values]
    
    for net in networks:
        net.store_patterns(patterns)
    
    test_pattern = patterns[0].clone()
    corruption_mask = torch.rand(N) < 0.2
    test_pattern[corruption_mask] *= -1
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (net, gamma) in enumerate(zip(networks, gamma_values)):
        x_init = test_pattern.unsqueeze(0)
        final_state, energy_hist = net.simulate_dynamics(x_init, steps=500)
        
        overlap = torch.dot(final_state.squeeze(), patterns[0]) / N
        
        axes[idx].plot(energy_hist, linewidth=2)
        axes[idx].set_xlabel('Time Steps (×10)')
        axes[idx].set_ylabel('Average Energy')
        axes[idx].set_title(f'Memory Retrieval (γ\' = {gamma})\nFinal Overlap: {overlap:.3f}')
        axes[idx].grid(True, alpha=0.3)
        
        if gamma < 0:
            axes[idx].plot(energy_hist, 'red', linewidth=2)
        else:
            axes[idx].plot(energy_hist, 'blue', linewidth=2)
    
    plt.tight_layout()
    plt.show()
    
    return networks, patterns

def main():
    print("Explosive Neural Networks - PyTorch Implementation")
    print("=" * 50)
    
    print("\n1. Analyzing explosive phase transitions...")
    beta_range = np.linspace(0.1, 3.0, 200)
    gamma_values = [0.0, -0.5, -1.0, -1.5]
    plot_phase_transition(beta_range, gamma_values)
    
    print("\n2. Demonstrating memory retrieval with explosive dynamics...")
    networks, patterns = demonstrate_memory_retrieval()
    
    print("\n3. Analyzing dynamical behavior...")
    
    N = 50
    steps = 1000
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    for gamma in [-1.5, -1.0, -0.5, 0.0]:
        net = ExplosiveNeuralNetwork(N, gamma, beta=1.5)
        patterns_single = create_binary_patterns(1, N, seed=123)
        net.store_patterns(patterns_single)
        
        x_init = patterns_single[0].clone().unsqueeze(0)
        noise = torch.randn_like(x_init) * 0.1
        x_init = torch.sign(x_init + noise)
        
        final_state, energy_hist = net.simulate_dynamics(x_init, steps)
        
        ax1.plot(energy_hist, label=f'γ\' = {gamma}', linewidth=2)
        
        beta_eff_hist = []
        x = x_init.clone()
        for i in range(0, steps, 10):
            x = net.glauber_step(x)
            beta_eff = net.effective_temperature(x).mean().item()
            beta_eff_hist.append(beta_eff)
        
        ax2.plot(beta_eff_hist, label=f'γ\' = {gamma}', linewidth=2)
    
    ax1.set_xlabel('Time Steps (×10)')
    ax1.set_ylabel('Average Energy')
    ax1.set_title('Energy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time Steps (×10)')
    ax2.set_ylabel('Effective Temperature β\'')
    ax2.set_title('Self-Regulated Annealing')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nKey observations:")
    print("- Negative γ' values exhibit explosive phase transitions with hysteresis")
    print("- Self-regulated annealing accelerates memory retrieval")
    print("- Higher-order interactions enhance memory capacity")
    print("- Curved activation functions enable novel collective behaviors")

if __name__ == "__main__":
    main()

