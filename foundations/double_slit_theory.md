# Double Slit Experiment - Mathematical Theory

## Course: Quantum Mechanics & Wave Optics

---

## 1. Introduction

The double slit experiment is a cornerstone demonstration in quantum mechanics that reveals the wave-particle duality of matter and light. This document presents the mathematical framework underlying the interference patterns observed in this fundamental experiment.

### Learning Objectives
- Understand wave interference principles
- Derive the interference pattern equations
- Apply Fraunhofer diffraction theory
- Calculate fringe spacing and intensity distributions

---

## 2. Wave Theory Fundamentals

### 2.1 Wave Equation

A monochromatic wave can be described by:

$$\Psi(r, t) = A e^{i(k \cdot r - \omega t)}$$

where:
- $A$ is the amplitude
- $k = \frac{2\pi}{\lambda}$ is the wave number
- $\omega = 2\pi f$ is the angular frequency
- $\lambda$ is the wavelength
- $r$ is the position vector

### 2.2 Principle of Superposition

When two or more waves overlap, the resultant amplitude is the sum of individual amplitudes:

$$\Psi_{total} = \Psi_1 + \Psi_2 + \cdots + \Psi_n$$

The intensity is proportional to the square of the amplitude:

$$I \propto |\Psi_{total}|^2$$

---

## 3. Geometry of the Double Slit

### 3.1 Experimental Setup

Consider two parallel slits separated by distance $d$, with a detection screen at distance $L$ from the slits.

**Coordinate System:**
- Origin at the midpoint between the slits
- Slits at positions $y = \pm\frac{d}{2}$
- Screen parallel to the slit plane at distance $L$

### 3.2 Path Difference

For a point $P$ at position $y$ on the screen:

$$r_1 = \sqrt{L^2 + (y + \frac{d}{2})^2}$$

$$r_2 = \sqrt{L^2 + (y - \frac{d}{2})^2}$$

The path difference is:

$$\Delta = r_2 - r_1$$

### 3.3 Far-Field Approximation

When $L \gg d$ and $L \gg y$ (Fraunhofer regime), we can use small angle approximations:

$$\sin\theta \approx \tan\theta \approx \frac{y}{L}$$

The path difference simplifies to:

$$\Delta \approx d \sin\theta = \frac{dy}{L}$$

---

## 4. Interference Conditions

### 4.1 Constructive Interference

Constructive interference occurs when the path difference is an integer multiple of the wavelength:

$$\Delta = n\lambda, \quad n = 0, \pm 1, \pm 2, \pm 3, \ldots$$

This gives bright fringes at positions:

$$y_{\text{bright}} = \frac{n\lambda L}{d}$$

### 4.2 Destructive Interference

Destructive interference occurs when the path difference is a half-integer multiple of the wavelength:

$$\Delta = (n + \frac{1}{2})\lambda, \quad n = 0, \pm 1, \pm 2, \ldots$$

This gives dark fringes at positions:

$$y_{\text{dark}} = \frac{(n + \frac{1}{2})\lambda L}{d}$$

### 4.3 Fringe Spacing

The distance between adjacent bright (or dark) fringes is:

$$\Delta y = \frac{\lambda L}{d}$$

**Key Observation:** The fringe spacing is:
- Proportional to wavelength $\lambda$ (longer wavelengths → wider spacing)
- Proportional to screen distance $L$
- Inversely proportional to slit separation $d$

---

## 5. Intensity Distribution

### 5.1 Two-Slit Interference (Ideal Case)

For two infinitesimally narrow slits, the intensity pattern is:

$$I(\theta) = I_0 \cos^2\left(\frac{\pi d \sin\theta}{\lambda}\right)$$

Or in terms of position $y$:

$$I(y) = I_0 \cos^2\left(\frac{\pi d y}{\lambda L}\right)$$

where $I_0$ is the maximum intensity.

### 5.2 Phase Difference

The phase difference between waves from the two slits is:

$$\delta = \frac{2\pi}{\lambda}\Delta = \frac{2\pi d \sin\theta}{\lambda}$$

The intensity can also be written as:

$$I = 4I_0 \cos^2\left(\frac{\delta}{2}\right)$$

---

## 6. Single Slit Diffraction

### 6.1 Diffraction from Finite Slit Width

Real slits have finite width $a$. The diffraction pattern from a single slit of width $a$ is:

$$I_{\text{single}}(\theta) = I_0 \left[\frac{\sin\beta}{\beta}\right]^2$$

where:

$$\beta = \frac{\pi a \sin\theta}{\lambda}$$

### 6.2 Minima Condition

The single slit diffraction minima occur when:

$$a \sin\theta = m\lambda, \quad m = \pm 1, \pm 2, \pm 3, \ldots$$

---

## 7. Combined Double Slit Pattern

### 7.1 Complete Intensity Formula

For two slits of finite width $a$ separated by distance $d$, the intensity is the product of the single-slit diffraction envelope and the double-slit interference pattern:

$$I(\theta) = I_0 \left[\frac{\sin\beta}{\beta}\right]^2 \cos^2\delta$$

where:
- $\beta = \frac{\pi a \sin\theta}{\lambda}$ (single slit diffraction parameter)
- $\delta = \frac{\pi d \sin\theta}{\lambda}$ (double slit interference parameter)

### 7.2 Expanded Form

$$I(\theta) = I_0 \left[\frac{\sin\left(\frac{\pi a \sin\theta}{\lambda}\right)}{\frac{\pi a \sin\theta}{\lambda}}\right]^2 \cos^2\left(\frac{\pi d \sin\theta}{\lambda}\right)$$

### 7.3 Physical Interpretation

- **First term** $\left[\frac{\sin\beta}{\beta}\right]^2$: Single slit diffraction envelope (modulates overall intensity)
- **Second term** $\cos^2\delta$: Interference pattern (creates fine fringes)

---

## 8. Quantum Mechanical Description

### 8.1 Probability Amplitude

In quantum mechanics, the wave function $\Psi$ represents a probability amplitude:

$$\Psi = \Psi_1 + \Psi_2$$

The probability of detection is:

$$P = |\Psi|^2 = |\Psi_1 + \Psi_2|^2 = |\Psi_1|^2 + |\Psi_2|^2 + 2\text{Re}(\Psi_1^*\Psi_2)$$

The last term is the **interference term**, which has no classical analog.

### 8.2 Feynman's Path Integral Approach

The probability amplitude for a particle to reach point $P$ is the sum over all possible paths:

$$\Psi(P) = \sum_{\text{all paths}} A_{\text{path}} e^{iS_{\text{path}}/\hbar}$$

where $S$ is the action along each path.

---

## 9. Matter Waves: de Broglie Wavelength

### 9.1 Wave-Particle Duality

Louis de Broglie proposed that particles have an associated wavelength:

$$\lambda = \frac{h}{p}$$

where:
- $h = 6.626 \times 10^{-34}$ J·s (Planck's constant)
- $p$ is the particle momentum

### 9.2 Electron Double Slit

For an electron accelerated through potential $V$:

$$p = \sqrt{2m_e eV}$$

$$\lambda_e = \frac{h}{\sqrt{2m_e eV}}$$

**Numerical Example:**
- Electron mass: $m_e = 9.109 \times 10^{-31}$ kg
- Accelerating voltage: $V = 50$ kV
- Resulting wavelength: $\lambda_e \approx 5.5$ pm

---

## 10. Visibility and Coherence

### 10.1 Fringe Visibility

The visibility (or contrast) of the interference pattern is defined as:

$$V = \frac{I_{\max} - I_{\min}}{I_{\max} + I_{\min}}$$

For perfect coherence and equal intensities: $V = 1$

### 10.2 Coherence Requirements

Clear interference requires:
1. **Temporal coherence**: monochromatic source ($\Delta\lambda \ll \lambda$)
2. **Spatial coherence**: slits illuminated coherently

The coherence length is:

$$L_c = \frac{\lambda^2}{\Delta\lambda}$$

---

## 11. Worked Examples

### Example 1: Visible Light Double Slit

**Given:**
- Wavelength: $\lambda = 632.8$ nm (He-Ne laser)
- Slit separation: $d = 0.3$ mm
- Screen distance: $L = 2$ m

**Find:** Fringe spacing

**Solution:**

$$\Delta y = \frac{\lambda L}{d} = \frac{(632.8 \times 10^{-9})(2)}{0.3 \times 10^{-3}}$$

$$\Delta y = 4.22 \times 10^{-3} \text{ m} = 4.22 \text{ mm}$$

---

### Example 2: Position of Third Bright Fringe

**Given:** Same as Example 1

**Find:** Position of $n = 3$ bright fringe

**Solution:**

$$y_3 = \frac{n\lambda L}{d} = \frac{3 \times 632.8 \times 10^{-9} \times 2}{0.3 \times 10^{-3}}$$

$$y_3 = 12.66 \text{ mm}$$

---

### Example 3: Missing Orders

**Given:**
- Slit width: $a = 0.1$ mm
- Slit separation: $d = 0.3$ mm

**Find:** Which interference maxima are missing?

**Solution:**

Missing orders occur when a diffraction minimum coincides with an interference maximum:

$$\frac{d}{a} = \frac{0.3}{0.1} = 3$$

Every 3rd order is missing: $n = 3, 6, 9, \ldots$

---

## 12. Advanced Topics

### 12.1 Multiple Slits (Diffraction Grating)

For $N$ slits, the intensity is:

$$I(\theta) = I_0 \left[\frac{\sin\beta}{\beta}\right]^2 \left[\frac{\sin(N\delta/2)}{\sin(\delta/2)}\right]^2$$

Principal maxima occur at:

$$d\sin\theta = n\lambda$$

### 12.2 Fresnel Diffraction

For the near-field (Fresnel regime where $L \sim d$), the full Fresnel integral must be used:

$$U(P) = \frac{i}{\lambda} \iint_{\text{aperture}} \frac{e^{ikr}}{r} \cos(\mathbf{n}, \mathbf{r}) \, dS$$

---

## 13. Experimental Considerations

### 13.1 Resolution Criteria

The Rayleigh criterion for resolution:

$$\theta_{\min} = 1.22\frac{\lambda}{D}$$

where $D$ is the aperture diameter.

### 13.2 Sources of Error

1. **Finite source size**: Reduces fringe visibility
2. **Chromatic dispersion**: Smears fringes for white light
3. **Mechanical vibrations**: Blurs the pattern
4. **Slit imperfections**: Distorts intensity distribution

---

## 14. Problem Set

### Problems for Practice

1. **Problem 1:** Calculate the angle to the first minimum in a single slit diffraction pattern for $\lambda = 550$ nm and $a = 0.1$ mm.

2. **Problem 2:** A double slit with $d = 0.2$ mm produces fringes 3 mm apart on a screen 1.5 m away. Find the wavelength.

3. **Problem 3:** Derive the condition for the first missing order in a double slit pattern.

4. **Problem 4:** Calculate the de Broglie wavelength of an electron moving at $10^6$ m/s.

5. **Problem 5:** Show that for two coherent sources of equal intensity, the maximum intensity is four times that of a single source.

---

## 15. Summary

### Key Equations

| Concept | Equation |
|---------|----------|
| Fringe spacing | $\Delta y = \frac{\lambda L}{d}$ |
| Bright fringes | $y_n = \frac{n\lambda L}{d}$ |
| Dark fringes | $y_n = \frac{(n+\frac{1}{2})\lambda L}{d}$ |
| Path difference | $\Delta = d\sin\theta$ |
| Double slit intensity | $I = I_0\cos^2\left(\frac{\pi d\sin\theta}{\lambda}\right)$ |
| Single slit envelope | $I = I_0\left[\frac{\sin\beta}{\beta}\right]^2$ |
| de Broglie wavelength | $\lambda = \frac{h}{p}$ |

### Physical Insights

1. **Wave Nature**: Interference demonstrates wave behavior
2. **Particle Nature**: Individual detection events show particle behavior
3. **Superposition**: Quantum amplitudes add, not classical probabilities
4. **Complementarity**: Wave and particle aspects are complementary
5. **Measurement**: Observation affects the quantum state

---

## References

1. Feynman, R.P., Leighton, R.B., & Sands, M. (1965). *The Feynman Lectures on Physics, Vol. III*
2. Griffiths, D.J. (2005). *Introduction to Quantum Mechanics* (2nd ed.)
3. Born, M., & Wolf, E. (1999). *Principles of Optics* (7th ed.)
4. Zeilinger, A. (1999). Experiment and the foundations of quantum physics. *Reviews of Modern Physics*, 71(2), S288-S297
5. Jönsson, C. (1961). Elektroneninterferenzen an mehreren künstlich hergestellten Feinspalten. *Zeitschrift für Physik*, 161(4), 454-474

---

## Appendix: Derivation of Intensity Formula

### Step-by-Step Derivation

**Step 1:** Wave from slit 1:
$$\Psi_1 = \frac{A}{r_1}e^{i(kr_1 - \omega t)}$$

**Step 2:** Wave from slit 2:
$$\Psi_2 = \frac{A}{r_2}e^{i(kr_2 - \omega t)}$$

**Step 3:** Total amplitude (for $r_1 \approx r_2 \approx r$):
$$\Psi = A\frac{e^{-i\omega t}}{r}(e^{ikr_1} + e^{ikr_2})$$

**Step 4:** Factor out average phase:
$$\Psi = A\frac{e^{-i\omega t}}{r}e^{ik(r_1+r_2)/2}(e^{ik(r_1-r_2)/2} + e^{-ik(r_1-r_2)/2})$$

**Step 5:** Use $r_1 - r_2 = -d\sin\theta$:
$$\Psi = 2A\frac{e^{-i\omega t}}{r}e^{ik\bar{r}}\cos\left(\frac{kd\sin\theta}{2}\right)$$

**Step 6:** Intensity:
$$I = |\Psi|^2 = 4I_0\cos^2\left(\frac{\pi d\sin\theta}{\lambda}\right)$$

where $I_0 = \frac{A^2}{r^2}$

---

*Document prepared for educational purposes | Last updated: January 5, 2026*
