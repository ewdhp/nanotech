# Fourier Analysis Demonstrations

## Basel Problem Proof via Parseval's Identity

### File: `parseval_basel_proof.py`

A complete demonstration showing how to prove that ∑(1/n²) = π²/6 using Fourier series and Parseval's identity.

### Mathematical Approach

1. **Function**: f(x) = x on the interval [-π, π]

2. **Fourier Series Expansion**:
   - Since f(x) is odd, only sine terms appear
   - Coefficients: bₙ = 2(-1)^(n+1)/n
   - Series: f(x) = ∑ bₙsin(nx)

3. **Parseval's Identity**:
   ```
   ∑|cₙ|² = (1/2π)∫₋π^π |f(x)|² dx
   ```

4. **Right-Hand Side**:
   ```
   (1/2π)∫₋π^π x² dx = π²/3
   ```

5. **Left-Hand Side**:
   - Complex coefficients: |cₙ|² = 1/n²
   - Sum: ∑_{n=-∞}^∞ |cₙ|² = 2∑_{n=1}^∞ 1/n²

6. **Result**:
   ```
   2∑_{n=1}^∞ 1/n² = π²/3
   
   Therefore: ∑_{n=1}^∞ 1/n² = π²/6 ✓
   ```

### Features

- **Analytical calculations** of Fourier coefficients
- **Numerical verification** via integration
- **Step-by-step proof** printed to console
- **8 comprehensive visualizations**:
  - Fourier series reconstruction with multiple term counts
  - Fourier coefficient plots
  - Parseval identity convergence
  - Basel sum convergence
  - Error analysis
  - Summary diagram

### Usage

```bash
# Install dependencies (if needed)
pip install numpy matplotlib seaborn scipy

# Run the demonstration
python3 parseval_basel_proof.py
```

### Output

The script displays:
1. Detailed step-by-step mathematical proof in the terminal
2. Numerical verification of each step
3. Interactive matplotlib figures showing all aspects of the proof

### Dependencies

- `numpy` — numerical computations
- `matplotlib` — plotting
- `seaborn` — styling
- `scipy` — numerical integration (quad function)

### Key Insights

- This proof is more elementary than Euler's infinite product method
- Parseval's identity directly connects L² norms to Fourier coefficients
- The sum of squares |cₙ|² immediately yields the Basel result
- Visual convergence shows how partial sums approach π²/6
