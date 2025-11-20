#!/usr/bin/env python3

def print_header(title):
    print("\n" + "="*60)
    print(title)
    print("="*60 + "\n")

def main():

    # -----------------------------------------------------------
    # 1. INTRODUCTION SUMMARY
    # -----------------------------------------------------------
    print_header("SUMMARY: Niels Bohr Introduction")

    intro = """
Niels Bohr (1885–1962) was a Danish physicist who fundamentally changed our 
understanding of atomic structure and quantum theory. 

In 1913, Bohr proposed a model of the hydrogen atom where:
- Electrons occupy *discrete* (quantized) orbits
- Only certain energy levels are allowed
- Electrons can jump between levels by absorbing/emitting a photon
- The frequency of emitted light matches the energy difference

The Bohr model successfully explained the spectral lines of hydrogen and 
introduced the idea of quantization into atomic physics.
"""
    print(intro)

    # -----------------------------------------------------------
    # 2. THEORY: BOHR MODEL FORMULAS
    # -----------------------------------------------------------
    print_header("BOHR MODEL THEORY AND EQUATIONS")

    theory = """
1. **Quantized Angular Momentum**
   L = n * ħ
   (Electrons can only have integer multiples of reduced Planck’s constant.)

2. **Bohr Radius (radius of nth orbit)**
   r_n = n^2 * a0
   where a0 = 5.29177 × 10⁻¹¹ m (Bohr radius)

3. **Energy Levels of Hydrogen**
   E_n = -13.6 eV / n²
   (Energy becomes less negative as n increases; n → ∞ gives ionization.)

4. **Orbital Velocity**
   v_n = (2.19 × 10⁶ m/s) / n

5. **Photon Emission / Absorption**
   When an electron transitions between levels:
   ΔE = E_f - E_i = h * f

6. **Rydberg Formula for Hydrogen Spectrum**
   1/λ = R * (1/n₁² - 1/n₂²)
   where R = 1.097 × 10⁷ m⁻¹ (Rydberg constant),

   This formula explains the Balmer series, Lyman series, etc.
"""
    print(theory)

    # -----------------------------------------------------------
    # 3. EXAMPLES CALCULATED BY PYTHON
    # -----------------------------------------------------------
    print_header("EXAMPLE: Computing Hydrogen Energy Levels")

    import math

    def energy_level(n):
        return -13.6 / (n**2)   # eV

    for n in range(1, 6):
        print(f"n = {n}:  E_n = {energy_level(n):.4f} eV")

    # -----------------------------------------------------------
    # 4. SAMPLE TRANSITION CALCULATION
    # -----------------------------------------------------------
    print_header("EXAMPLE: Photon Emitted (Transition n=3 → n=2)")

    E3 = energy_level(3)
    E2 = energy_level(2)
    delta_E = E2 - E3  # emission => positive energy photon

    h = 4.135667e-15      # eV·s
    c = 3e8               # m/s
    f = delta_E / h
    wavelength = c / f

    print(f"Energy difference: {delta_E:.4f} eV")
    print(f"Frequency:         {f:.3e} Hz")
    print(f"Wavelength:        {wavelength*1e9:.2f} nm  (visible: Balmer series)")

    print("\nDone.")

if __name__ == "__main__":
    main()
