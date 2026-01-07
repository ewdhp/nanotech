# VS Code + Jupyter + SageMath Setup Guide

## ✅ What You Have Already

- ✓ SageMath 10.7 installed in conda environment `sage`
- ✓ Jupyter already installed (jupyterlab 4.5.1)
- ✓ ipywidgets installed (needed for interactive sliders)

## 📦 What You Need to Install

### 1. VS Code Extension (Required)

**Install the Jupyter extension:**

```
Extension ID: ms-toolsai.jupyter
```

**How to install:**
- Open VS Code Extensions (Ctrl+Shift+X)
- Search for "Jupyter"
- Install "Jupyter" by Microsoft (the main one)

Or click the "Install" button above if you see it!

---

## 🚀 How to Use

### Method 1: Open Notebook in VS Code (Easiest)

1. **Open the notebook file:**
   - In VS Code, open: `test_vscode_interactive.ipynb`
   - VS Code will automatically detect it's a Jupyter notebook

2. **Select the SageMath kernel:**
   - Click "Select Kernel" in the top-right
   - Choose "Python Environments..."
   - Select: `/home/ewd/miniconda3/envs/sage/bin/python`

3. **Run the cells:**
   - Click the ▶️ play button on each cell
   - You should see **sliders** appear!

### Method 2: Command Palette

1. Press `Ctrl+Shift+P`
2. Type: "Create: New Jupyter Notebook"
3. Select kernel: `sage` environment
4. Start coding!

---

## 📝 Test Files Created

1. **[test_vscode_interactive.ipynb](test_vscode_interactive.ipynb)**
   - Quick test with 3 examples
   - Open this first to verify everything works!

2. **[sage_interactive_plots.ipynb](sage_interactive_plots.ipynb)**
   - Full collection of 7 interactive examples
   - Open this after the test works

---

## 🎯 Quick Test

Open `test_vscode_interactive.ipynb` in VS Code and run the cells. You should see:

1. ✓ SageMath imports successfully
2. 🎚️ Sliders appear above plots
3. 📊 Plots update when you move sliders

---

## 🐛 Troubleshooting

### Issue: "No kernel found"
**Solution:** 
- Make sure you selected the sage Python interpreter
- Path: `/home/ewd/miniconda3/envs/sage/bin/python`

### Issue: "ipywidgets not found"
**Solution:**
```bash
conda activate sage
conda install ipywidgets -y
```

### Issue: Sliders don't appear
**Solution:**
- Make sure you have the `@interact` decorator
- Import: `from sage.repl.ipython_kernel.interact import interact`
- Restart the kernel: Click "Restart" button in notebook toolbar

### Issue: Plots don't display
**Solution:**
- Use `show(p)` instead of just `p`
- Or use `p.show()` for inline display

---

## 💡 Comparison: VS Code vs Browser Jupyter

| Feature | VS Code | Browser Jupyter |
|---------|---------|-----------------|
| Editor | Full VS Code features | Basic web editor |
| Git integration | ✓ Built-in | ✗ Manual |
| Debugging | ✓ Full debugger | Limited |
| Extensions | ✓ All VS Code extensions | Jupyter extensions only |
| Multiple files | ✓ Easy | New tabs |
| Interactive sliders | ✓ Works great | ✓ Works great |
| Performance | Same | Same |

**Recommendation:** Use VS Code for development, browser for quick experiments.

---

## 🎓 Next Steps

1. ✅ Install the Jupyter extension in VS Code
2. ✅ Open `test_vscode_interactive.ipynb`
3. ✅ Select sage kernel
4. ✅ Run cells and verify sliders work
5. ✅ Explore `sage_interactive_plots.ipynb` for more examples
6. 🎉 Start creating your own interactive plots!

---

## 🔗 Resources

- [VS Code Jupyter Documentation](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)
- [SageMath Documentation](https://doc.sagemath.org/)
- [Interactive Plots Guide](INTERACTIVE_PLOTS_GUIDE.md)

**You're all set!** 🚀
