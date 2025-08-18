"""
Root conftest.py for pytest configuration.
Automatically adds all package src directories to Python path using package-discovery.sh.
"""
import sys
import subprocess
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent

# Use the existing package-discovery.sh to get all packages
package_discovery_script = PROJECT_ROOT / "bin" / "package-discovery.sh"

if package_discovery_script.exists():
    try:
        # Run package-discovery.sh to get list of packages
        result = subprocess.run(
            [str(package_discovery_script), "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the package names
        packages = result.stdout.strip().split()
        
        # Add each package's src directory to Python path
        for package_name in packages:
            src_dir = PROJECT_ROOT / "packages" / package_name / "src"
            if src_dir.exists():
                src_path = str(src_dir.absolute())
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                    print(f"Added to Python path: {package_name}/src")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to run package-discovery.sh: {e}")
        # Fallback to manual discovery
        packages_dir = PROJECT_ROOT / "packages"
        if packages_dir.exists():
            for package_dir in packages_dir.iterdir():
                if package_dir.is_dir():
                    src_dir = package_dir / "src"
                    if src_dir.exists():
                        src_path = str(src_dir.absolute())
                        if src_path not in sys.path:
                            sys.path.insert(0, src_path)

# Also add the project root for any top-level imports
root_path = str(PROJECT_ROOT.absolute())
if root_path not in sys.path:
    sys.path.insert(0, root_path)