"""
Max Planck: Historical Context and Leadership

Timeline of key events:
- 1858: Born in Kiel, Germany
- 1879: PhD in thermodynamics (Munich)
- 1900: Quantum hypothesis announced (E = hν)
- 1918: Nobel Prize in Physics
- 1930-1937: President of Kaiser Wilhelm Society
- 1947: Died in Göttingen; society renamed Max Planck Society

Leadership:
- Guided German physics through WWI and WWII.
- Advocated for international scientific cooperation.
- Mentored many physicists who shaped quantum mechanics.

Demonstration:
- Timeline visualization.
- Impact network (Planck and his contemporaries).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np


def main():
    # Timeline data
    events = [
        (1858, "Born in Kiel"),
        (1879, "PhD (Munich)"),
        (1900, "Quantum hypothesis"),
        (1918, "Nobel Prize"),
        (1930, "President, Kaiser Wilhelm Society"),
        (1947, "Died; society renamed Max Planck Society"),
    ]

    years = [e[0] for e in events]
    labels = [e[1] for e in events]

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # (a) Timeline
    axes[0].plot(years, np.zeros_like(years), 'ko', markersize=10)
    for i, (year, label) in enumerate(events):
        offset = 0.5 if i % 2 == 0 else -0.5
        axes[0].text(year, offset, f"{year}\n{label}", ha='center', va='bottom' if offset > 0 else 'top',
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        axes[0].plot([year, year], [0, offset], 'k--', alpha=0.5)

    axes[0].set_ylim(-1.5, 1.5)
    axes[0].set_xlim(1850, 1955)
    axes[0].set_xlabel("Year")
    axes[0].set_yticks([])
    axes[0].set_title("Max Planck: Timeline of Key Events")
    axes[0].grid(True, axis='x', alpha=0.3)

    # (b) Influence network (simplified)
    # Nodes: Planck and contemporaries
    names = ["Max Planck", "Einstein", "Bohr", "Heisenberg", "Schrödinger", "Pauli"]
    positions = {
        "Max Planck": (0, 0),
        "Einstein": (-1, 1),
        "Bohr": (1, 1),
        "Heisenberg": (-1, -1),
        "Schrödinger": (1, -1),
        "Pauli": (0, -1.5),
    }

    connections = [
        ("Max Planck", "Einstein"),
        ("Max Planck", "Bohr"),
        ("Max Planck", "Heisenberg"),
        ("Max Planck", "Schrödinger"),
        ("Einstein", "Bohr"),
        ("Bohr", "Heisenberg"),
        ("Heisenberg", "Pauli"),
        ("Schrödinger", "Pauli"),
    ]

    for (n1, n2) in connections:
        x1, y1 = positions[n1]
        x2, y2 = positions[n2]
        axes[1].plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=1)

    for name, (x, y) in positions.items():
        color = 'gold' if name == "Max Planck" else 'lightgreen'
        size = 1500 if name == "Max Planck" else 800
        axes[1].scatter(x, y, s=size, c=color, edgecolors='black', linewidth=2, zorder=5)
        axes[1].text(x, y, name, ha='center', va='center', fontsize=10, weight='bold')

    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].set_aspect('equal')
    axes[1].axis('off')
    axes[1].set_title("Planck's Influence Network (simplified)\nCentral role in early quantum physics")

    plt.tight_layout()
    plt.show()

    print("=== Max Planck: Historical Context and Leadership ===\n")
    print("Timeline:")
    for year, label in events:
        print(f"  {year}: {label}")
    print()
    print("Leadership and Legacy:")
    print("  - President of Kaiser Wilhelm Society (1930-1937)")
    print("  - Mentored and influenced Einstein, Bohr, Heisenberg, and others")
    print("  - Society renamed 'Max Planck Society' in 1948, now Germany's premier research organization")
    print("  - His quantum hypothesis laid the foundation for quantum mechanics\n")


if __name__ == "__main__":
    main()
