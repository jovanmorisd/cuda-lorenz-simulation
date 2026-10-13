"""
Verification script for AI-HPC Lorenz Project
Checks all dependencies and displays system information
"""

import sys
import os

def check_python():
    """Check Python version"""
    print("Python Information:")
    print(f"  Version: {sys.version.split()[0]}")
    print(f"  Executable: {sys.executable}")
    return True

def check_packages():
    """Check required packages"""
    print("\nPackage Status:")
    
    packages = {
        'numpy': 'NumPy',
        'torch': 'PyTorch',
        'matplotlib': 'Matplotlib',
        'sklearn': 'scikit-learn',
        'jupyter': 'Jupyter',
    }
    
    all_ok = True
    for pkg, name in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, '__version__', 'unknown')
            print(f"  ✓ {name:15s} {version}")
        except ImportError:
            print(f"  ✗ {name:15s} NOT INSTALLED")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Check CUDA availability"""
    print("\nGPU/CUDA Information:")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            print(f"  ✓ CUDA Available: {cuda_available}")
            print(f"  ✓ CUDA Version: {cuda_version}")
            print(f"  ✓ Device Count: {device_count}")
            print(f"  ✓ Device 0: {device_name}")
            return True
        else:
            print(f"  ⚠ CUDA Not Available (CPU mode)")
            return False
    except Exception as e:
        print(f"  ✗ Error checking CUDA: {e}")
        return False

def check_compilers():
    """Check C++ and CUDA compilers"""
    import subprocess
    
    print("\nCompiler Status:")
    
    # Check g++
    try:
        result = subprocess.run(['g++', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"  ✓ g++: {version}")
        else:
            print(f"  ✗ g++ not found")
    except FileNotFoundError:
        print(f"  ✗ g++ not found")
    
    # Check OpenMP
    try:
        result = subprocess.run(['g++', '-fopenmp', '-dM', '-E', '-'], 
                              input='', capture_output=True, text=True)
        if '_OPENMP' in result.stdout:
            print(f"  ✓ OpenMP: Available")
        else:
            print(f"  ⚠ OpenMP: Not available")
    except:
        print(f"  ⚠ OpenMP: Could not determine")
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[-2]
            print(f"  ✓ nvcc: {version}")
        else:
            print(f"  ⚠ nvcc not found (GPU support disabled)")
    except FileNotFoundError:
        print(f"  ⚠ nvcc not found (GPU support disabled)")

def check_directories():
    """Check project directory structure"""
    print("\nProject Structure:")
    
    required_dirs = ['src', 'cuda', 'data', 'models', 'plots', 'report']
    required_files = ['lorenz_hpc.ipynb', 'README.md', 'requirements.txt']
    
    for d in required_dirs:
        if os.path.isdir(d):
            print(f"  ✓ {d}/ exists")
        else:
            print(f"  ✗ {d}/ missing")
    
    for f in required_files:
        if os.path.isfile(f):
            print(f"  ✓ {f} exists")
        else:
            print(f"  ✗ {f} missing")

def main():
    print("="*50)
    print("AI-HPC Lorenz Project - System Verification")
    print("="*50)
    
    check_python()
    packages_ok = check_packages()
    check_cuda()
    check_compilers()
    check_directories()
    
    print("\n" + "="*50)
    if packages_ok:
        print("✓ All critical packages installed!")
        print("\nYou can now run: jupyter notebook lorenz_hpc.ipynb")
    else:
        print("⚠ Some packages missing. Install with:")
        print("  pip install -r requirements.txt")
    print("="*50)

if __name__ == '__main__':
    main()
