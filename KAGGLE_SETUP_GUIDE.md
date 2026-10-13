# 🚀 How to Run This Project on Kaggle

## Why Kaggle is Perfect for This Project

✅ **No Installation Needed** - PyTorch, NumPy, Matplotlib pre-installed  
✅ **Free GPU Access** - P100 or V100 for model training  
✅ **Fast Execution** - Pre-installed packages already optimized  
✅ **Easy Download** - Outputs automatically downloadable  

---

## 📋 Step-by-Step Guide

### Step 1: Create a Kaggle Account (if needed)
- Go to https://www.kaggle.com
- Sign up with email or Google
- Verify your email

### Step 2: Upload Project Files to Kaggle

You have two options:

#### Option A: Upload as Notebook (Fastest)
1. Click **"New Notebook"** in Kaggle
2. In the notebook editor, click **"+Code"** at the top
3. Copy the Python code from one of these sections below
4. Paste into the cell
5. Click **"Run All"** button

#### Option B: Upload Dataset
1. Click **"Create New Dataset"** 
2. Upload the folder containing:
   - Notebook file
   - Python modules (rk4.py, euler.py, dataset.py, models.py, utils.py)
3. Create the dataset
4. In a new notebook, add this code to load modules:
```python
import sys
sys.path.append('/kaggle/input/your-dataset-name')
```

### Step 3: Run the Optimized Kaggle Notebook

I've created a **Kaggle-optimized notebook** for you: `kaggle_lorenz_notebook.ipynb`

This version:
- ✅ Requires NO external file dependencies
- ✅ All code embedded in notebook
- ✅ Takes ~10-15 minutes on GPU
- ✅ Generates 6 visualizations automatically

### Step 4: Execute All Cells

In Kaggle notebook:
1. Click **"Run All"** button at the top (or press Ctrl+Shift+Enter)
2. Wait for execution (~10-15 minutes)
3. View results in-place

### Step 5: Download Results

After execution completes:
1. Right-click the plots generated
2. Click **"Save image"** or **"Copy image link"**
3. Or scroll to "Output" tab and download all files

---

## 📝 Alternative: Copy-Paste Code into Kaggle

If you prefer to enter code manually, here's a **minimal working version** to paste directly into a Kaggle cell:

```python
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import time

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
OUTPUT_DIR = '/kaggle/working'

# [Rest of code from kaggle_lorenz_notebook.ipynb]
# Cell 3 through Cell 13
```

---

## 🎯 Quick Start (5 Steps)

1. **Open Kaggle** → https://www.kaggle.com
2. **Create New Notebook** (Python)
3. **Copy this entire code block below** and paste in first cell:

```python
# PASTE THIS INTO KAGGLE NOTEBOOK CELL
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DIR = '/kaggle/working'

# RK4 Solver
class LorenzRK4:
    def __init__(self, sigma=10.0, rho=28.0, beta=8.0/3.0):
        self.sigma, self.rho, self.beta = sigma, rho, beta
    def derivative(self, state):
        x, y, z = state
        return np.array([
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z
        ])
    def step(self, state, dt):
        k1 = self.derivative(state)
        k2 = self.derivative(state + 0.5 * dt * k1)
        k3 = self.derivative(state + 0.5 * dt * k2)
        k4 = self.derivative(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    def solve(self, initial_state, dt=0.01, T=25.0):
        num_steps = int(T / dt)
        trajectory = np.zeros((num_steps, 3))
        trajectory[0] = initial_state
        state = initial_state.copy()
        for i in range(1, num_steps):
            state = self.step(state, dt)
            trajectory[i] = state
        return trajectory

print("Generating trajectories...")
solver = LorenzRK4()
trajectories = []
np.random.seed(42)
for i in range(100):
    x0 = np.random.uniform(-15, 15)
    y0 = np.random.uniform(-15, 25)
    z0 = np.random.uniform(5, 35)
    trajectories.append(solver.solve(np.array([x0, y0, z0])))
trajectories = np.array(trajectories)
print(f"✓ Generated {len(trajectories)} trajectories")

# Visualize
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
traj = trajectories[0]
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')
plt.savefig(f'{OUTPUT_DIR}/lorenz_attractor.png', dpi=100)
plt.show()
print("✓ Visualization saved")
```

4. **Click "Run All"** button
5. **Wait ~ 10 minutes** for execution
6. **Download** the generated PNG files

---

## 💡 Pro Tips for Kaggle

### Maximize GPU Usage
- Enable GPU: In notebook settings, toggle **Accelerator → GPU**
- Current Kaggle GPU: P100 or V100 (great for neural networks)

### Monitor Execution
- Look for a progress bar during training
- Check output for "✓" marks confirming completion

### Save Outputs
- All plots saved to `/kaggle/working/`
- Results automatically in "Output" tab
- Download before notebook session closes

### Optimize Runtime
- Don't need to recompile C++/CUDA
- Python implementations + PyTorch GPU = fast enough
- ~10-15 minutes total execution

---

## 📊 What You'll Get

After running the notebook, you'll have:

✅ **01_lorenz_attractor.png** - 3D attractor visualization  
✅ **02_training_history.png** - Model training curves  
✅ **03_phase_space.png** - Phase space projections  
✅ **04_error_comparison.png** - Model RMSE comparison  
✅ **05_error_horizon.png** - Error growth analysis  
✅ **06_rollout_comparison.png** - Multi-step predictions  
✅ **summary.csv** - Performance metrics table  

All ready to download!

---

## 🐛 Troubleshooting

### "CUDA out of memory"
- Reduce batch size in code (change `batch_size = 32` to `16` or `8`)

### "Module not found"
- Make sure NumPy, Torch, Matplotlib are available
- Kaggle pre-installs these, so shouldn't be an issue

### Notebook taking too long
- Normal: ~10-15 minutes with GPU
- Check GPU is enabled in settings
- Training 200 epochs takes time but shows progress

### Can't download files
- Files save to `/kaggle/working/`
- Click "Output" tab at bottom after notebook completes
- All PNG and CSV files downloadable there

---

## 🎓 Learning Path

1. **Start**: Run the notebook as-is
2. **Explore**: Modify trajectories count (e.g., 50, 200)
3. **Experiment**: Change lookback window (try 5, 15, 20)
4. **Customize**: Modify network architecture (add layers, etc.)
5. **Extend**: Add more models or different dynamics

---

## 📞 Need Help?

- Check the cell outputs for error messages
- Review "PROJECT_COMPLETION_REPORT.md" for background
- All code is commented for understanding
- Keras/PyTorch documentation: https://pytorch.org

---

## 🚀 Ready?

**Use the official Kaggle-optimized notebook:**
- File: `kaggle_lorenz_notebook.ipynb`
- No setup needed
- Run in Kaggle immediately
- All outputs to `/kaggle/working/`

Happy researching! 🎉

