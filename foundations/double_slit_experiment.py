"""
Double Slit Experiment Simulation
==================================

This script simulates the famous double slit experiment, demonstrating
wave-particle duality and quantum interference patterns.

The double slit experiment shows that light and matter can display characteristics
of both classically defined waves and particles. When particles pass through two
slits, they create an interference pattern characteristic of waves.

Key Physics:
- Wave interference: constructive and destructive interference
- Path difference: Δ = d·sin(θ) where d is slit separation
- Constructive interference: Δ = n·λ (n = 0, ±1, ±2, ...)
- Destructive interference: Δ = (n + 1/2)·λ
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class DoubleSlitExperiment:
    """
    Simulates the double slit experiment with quantum interference.
    """
    
    def __init__(self, wavelength=500e-9, slit_separation=1e-4, 
                 slit_width=2e-5, screen_distance=1.0):
        """
        Initialize the double slit experiment parameters.
        
        Parameters:
        -----------
        wavelength : float
            Wavelength of the light/particle (meters), default 500nm (green light)
        slit_separation : float
            Distance between slit centers (meters), default 0.1mm
        slit_width : float
            Width of each slit (meters), default 0.02mm
        screen_distance : float
            Distance from slits to detection screen (meters), default 1m
        """
        self.wavelength = wavelength
        self.d = slit_separation  # slit separation
        self.a = slit_width       # slit width
        self.L = screen_distance  # screen distance
        self.k = 2 * np.pi / wavelength  # wave number
        
    def single_slit_amplitude(self, y, slit_position):
        """
        Calculate the amplitude from a single slit using Fraunhofer diffraction.
        
        Parameters:
        -----------
        y : array
            Positions on the screen
        slit_position : float
            Position of the slit center
            
        Returns:
        --------
        complex array : Amplitude at each screen position
        """
        # Path difference for single slit diffraction
        r = np.sqrt(self.L**2 + (y - slit_position)**2)
        
        # Single slit diffraction pattern (Fraunhofer approximation)
        beta = (np.pi * self.a / self.wavelength) * (y - slit_position) / self.L
        
        # Avoid division by zero
        amplitude = np.ones_like(beta, dtype=complex)
        mask = np.abs(beta) > 1e-10
        amplitude[mask] = np.sin(beta[mask]) / beta[mask]
        
        # Add phase factor for propagation
        amplitude *= np.exp(1j * self.k * r) / r
        
        return amplitude
    
    def interference_pattern(self, y_positions):
        """
        Calculate the interference pattern on the screen.
        
        Parameters:
        -----------
        y_positions : array
            Positions on the screen where intensity is calculated
            
        Returns:
        --------
        array : Intensity at each position
        """
        # Positions of the two slits
        slit1_pos = -self.d / 2
        slit2_pos = self.d / 2
        
        # Amplitudes from each slit
        amplitude1 = self.single_slit_amplitude(y_positions, slit1_pos)
        amplitude2 = self.single_slit_amplitude(y_positions, slit2_pos)
        
        # Total amplitude (coherent superposition)
        total_amplitude = amplitude1 + amplitude2
        
        # Intensity is proportional to |amplitude|^2
        intensity = np.abs(total_amplitude)**2
        
        return intensity
    
    def analytical_pattern(self, y_positions):
        """
        Calculate interference pattern using analytical formula (far-field approximation).
        
        I(θ) ∝ [sin(β)/β]^2 · cos^2(δ)
        where β = (π·a·sin(θ))/λ and δ = (π·d·sin(θ))/λ
        """
        # Small angle approximation: sin(θ) ≈ tan(θ) ≈ y/L
        theta = np.arctan(y_positions / self.L)
        
        # Single slit diffraction factor
        beta = (np.pi * self.a * np.sin(theta)) / self.wavelength
        single_slit = np.ones_like(beta)
        mask = np.abs(beta) > 1e-10
        single_slit[mask] = (np.sin(beta[mask]) / beta[mask])**2
        
        # Double slit interference factor
        delta = (np.pi * self.d * np.sin(theta)) / self.wavelength
        double_slit = np.cos(delta)**2
        
        # Combined pattern
        intensity = single_slit * double_slit
        
        return intensity
    
    def fringe_spacing(self):
        """
        Calculate the fringe spacing (distance between bright fringes).
        
        Returns:
        --------
        float : Fringe spacing in meters
        """
        return self.wavelength * self.L / self.d
    
    def plot_pattern(self, method='numerical', num_points=1000):
        """
        Plot the interference pattern.
        
        Parameters:
        -----------
        method : str
            'numerical' or 'analytical'
        num_points : int
            Number of points to calculate
        """
        # Screen positions (±5cm from center)
        y = np.linspace(-0.05, 0.05, num_points)
        
        if method == 'numerical':
            intensity = self.interference_pattern(y)
            title = "Double Slit Experiment - Numerical Simulation"
        else:
            intensity = self.analytical_pattern(y)
            title = "Double Slit Experiment - Analytical Solution"
        
        # Normalize intensity
        intensity = intensity / np.max(intensity)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot 1: Intensity vs position
        ax1.plot(y * 1000, intensity, 'b-', linewidth=2)
        ax1.set_xlabel('Position on Screen (mm)', fontsize=12)
        ax1.set_ylabel('Normalized Intensity', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # Add fringe spacing annotation
        fringe_spacing = self.fringe_spacing()
        ax1.axvline(fringe_spacing * 1000, color='r', linestyle='--', 
                   alpha=0.5, label=f'Fringe spacing: {fringe_spacing*1000:.2f} mm')
        ax1.axvline(-fringe_spacing * 1000, color='r', linestyle='--', alpha=0.5)
        ax1.legend()
        
        # Plot 2: 2D intensity pattern
        # Create a 2D array by repeating the pattern
        pattern_2d = np.tile(intensity, (100, 1))
        extent = [y[0] * 1000, y[-1] * 1000, 0, 10]
        
        im = ax2.imshow(pattern_2d, aspect='auto', extent=extent, 
                       cmap='hot', origin='lower', interpolation='bilinear')
        ax2.set_xlabel('Position on Screen (mm)', fontsize=12)
        ax2.set_ylabel('Screen Height (mm)', fontsize=12)
        ax2.set_title('2D Interference Pattern', fontsize=12)
        plt.colorbar(im, ax=ax2, label='Normalized Intensity')
        
        # Add experiment parameters as text
        params_text = (
            f"Wavelength: {self.wavelength*1e9:.0f} nm\n"
            f"Slit separation: {self.d*1e6:.1f} μm\n"
            f"Slit width: {self.a*1e6:.1f} μm\n"
            f"Screen distance: {self.L:.2f} m\n"
            f"Fringe spacing: {fringe_spacing*1000:.2f} mm"
        )
        
        fig.text(0.02, 0.98, params_text, fontsize=10, 
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def compare_wavelengths(self):
        """
        Compare interference patterns for different wavelengths (colors).
        """
        wavelengths = {
            'Violet': 400e-9,
            'Blue': 450e-9,
            'Green': 550e-9,
            'Red': 650e-9
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        y = np.linspace(-0.05, 0.05, 1000)
        
        for idx, (color_name, wavelength) in enumerate(wavelengths.items()):
            # Temporarily change wavelength
            original_wavelength = self.wavelength
            self.wavelength = wavelength
            self.k = 2 * np.pi / wavelength
            
            intensity = self.analytical_pattern(y)
            intensity = intensity / np.max(intensity)
            
            # Restore original wavelength
            self.wavelength = original_wavelength
            self.k = 2 * np.pi / original_wavelength
            
            # Plot
            color_map = {'Violet': 'purple', 'Blue': 'blue', 
                        'Green': 'green', 'Red': 'red'}
            
            axes[idx].plot(y * 1000, intensity, 
                          color=color_map[color_name], linewidth=2)
            axes[idx].set_xlabel('Position (mm)', fontsize=11)
            axes[idx].set_ylabel('Intensity', fontsize=11)
            axes[idx].set_title(f'{color_name} Light (λ = {wavelength*1e9:.0f} nm)', 
                               fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_ylim([0, 1.1])
        
        plt.suptitle('Double Slit Patterns for Different Wavelengths', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return fig


def demonstrate_wave_particle_duality():
    """
    Demonstrate wave-particle duality with the double slit experiment.
    Shows how particles create wave-like interference patterns.
    """
    print("=" * 70)
    print("DOUBLE SLIT EXPERIMENT - Wave-Particle Duality Demonstration")
    print("=" * 70)
    print()
    print("The double slit experiment is one of the most important experiments")
    print("in quantum mechanics. It demonstrates that matter and light exhibit")
    print("both wave and particle properties.")
    print()
    print("Key Observations:")
    print("1. Particles pass through slits ONE AT A TIME")
    print("2. Each particle hits screen at ONE LOCATION (particle behavior)")
    print("3. Over time, particles build up INTERFERENCE PATTERN (wave behavior)")
    print("4. The pattern cannot be explained by classical particles alone")
    print()
    print("Mathematical Description:")
    print("- Constructive interference: d·sin(θ) = n·λ")
    print("- Destructive interference: d·sin(θ) = (n + 1/2)·λ")
    print("- Fringe spacing: Δy = λ·L/d")
    print()
    
    # Create experiment with visible light (green laser)
    experiment = DoubleSlitExperiment(
        wavelength=532e-9,      # 532 nm (green laser)
        slit_separation=0.2e-3, # 0.2 mm
        slit_width=0.03e-3,     # 0.03 mm
        screen_distance=2.0     # 2 meters
    )
    
    print(f"Experiment Parameters:")
    print(f"  Wavelength: {experiment.wavelength*1e9:.1f} nm (green light)")
    print(f"  Slit separation: {experiment.d*1e3:.2f} mm")
    print(f"  Slit width: {experiment.a*1e3:.3f} mm")
    print(f"  Screen distance: {experiment.L:.1f} m")
    print(f"  Expected fringe spacing: {experiment.fringe_spacing()*1000:.2f} mm")
    print()
    print("Generating interference pattern...")
    print()
    
    # Plot the pattern
    experiment.plot_pattern(method='analytical')
    
    # Compare different wavelengths
    print("Comparing different wavelengths...")
    experiment.compare_wavelengths()


def electron_double_slit():
    """
    Simulate double slit experiment with electrons (matter waves).
    Demonstrates de Broglie wavelength and wave nature of matter.
    """
    print("=" * 70)
    print("ELECTRON DOUBLE SLIT EXPERIMENT")
    print("=" * 70)
    print()
    print("Electrons, despite being particles with mass, also exhibit wave")
    print("behavior through the double slit experiment.")
    print()
    
    # Electron parameters
    m_e = 9.109e-31  # electron mass (kg)
    V = 50e3         # accelerating voltage (50 kV)
    e = 1.602e-19    # electron charge (C)
    h = 6.626e-34    # Planck constant
    
    # Calculate electron momentum and de Broglie wavelength
    # For non-relativistic: E = (1/2)mv² = eV
    # p = mv = √(2meV)
    # λ = h/p
    p = np.sqrt(2 * m_e * e * V)
    lambda_e = h / p
    
    print(f"Accelerating voltage: {V/1000:.0f} kV")
    print(f"Electron momentum: {p:.3e} kg·m/s")
    print(f"de Broglie wavelength: {lambda_e*1e12:.2f} pm")
    print()
    
    # Create experiment with electron wavelength
    experiment = DoubleSlitExperiment(
        wavelength=lambda_e,
        slit_separation=1e-6,   # 1 micrometer
        slit_width=0.2e-6,      # 0.2 micrometers
        screen_distance=0.5     # 0.5 meters
    )
    
    print(f"Slit separation: {experiment.d*1e6:.2f} μm")
    print(f"Slit width: {experiment.a*1e6:.2f} μm")
    print(f"Expected fringe spacing: {experiment.fringe_spacing()*1e3:.3f} mm")
    print()
    
    # Plot the pattern
    experiment.plot_pattern(method='analytical')


if __name__ == "__main__":
    # Run the main demonstration
    demonstrate_wave_particle_duality()
    
    # Uncomment to see electron double slit
    # electron_double_slit()
    
    print("\n" + "=" * 70)
    print("Quantum Interpretation:")
    print("=" * 70)
    print("""
The interference pattern persists even when particles are sent one at a time,
suggesting that each particle interferes with itself! This phenomenon cannot
be explained by classical physics and is fundamental to quantum mechanics.

Key insights:
1. Each particle goes through BOTH slits simultaneously (superposition)
2. The wave function describes probability amplitudes
3. Observation/measurement collapses the wave function
4. When we measure which slit the particle goes through, the interference
   pattern disappears (wave function collapse)

This experiment demonstrates the Copenhagen interpretation and the
fundamental role of measurement in quantum mechanics.
    """)
