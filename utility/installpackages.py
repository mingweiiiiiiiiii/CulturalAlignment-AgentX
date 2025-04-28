import ast
import os
import sys
from typing import Dict, List, Optional, Set

import pkg_resources
import stdlib_list


def extract_imports_from_file(filepath: str) -> Set[str]:
    """
    Extracts imported module names from a single Python file,
    including nested imports like 'from x.y import z'.
    """
    with open(filepath, "r", encoding="utf-8") as file:
        try:
            node = ast.parse(file.read(), filename=filepath)
        except SyntaxError:
            print(f"Warning: Skipping {filepath} (syntax error)")
            return set()

    imports = set()
    for item in ast.walk(node):
        if isinstance(item, ast.Import):
            for alias in item.names:
                # Get the top-level module name
                module_name = alias.name.split(".")[0]
                imports.add(module_name)
        elif isinstance(item, ast.ImportFrom):
            if item.module:
                # Handle cases like "from x.y import z"
                if item.level == 0:  # absolute import
                    # Get the top-level module
                    module_name = item.module.split(".")[0]
                    imports.add(module_name)

                    # Also add the full module path for better matching
                    imports.add(item.module)
    return imports


def get_package_mapping() -> Dict[str, str]:
    """Returns a mapping of module names to package names for cases where they differ."""
    return {
        "sklearn": "scikit-learn",
        "PIL": "pillow",
        "cv2": "opencv-python",
        "bs4": "beautifulsoup4",
        "yaml": "pyyaml",
        "dotenv": "python-dotenv",
        "google.generativeai": "google-generativeai",
        "genai": "google-generativeai",
        "mistralai": "mistralai",
        "langgraph": "langgraph",
        "langchain": "langchain",
        "langchain_core": "langchain-core",
        "langchain_groq": "langchain-groq",
    }


def get_installed_packages() -> Dict[str, pkg_resources.Distribution]:
    """Gets installed package information (name and distribution)."""
    return {pkg.key: pkg for pkg in pkg_resources.working_set}


def get_standard_libraries() -> Set[str]:
    """Gets a set of standard libraries based on Python version."""
    std_libs = stdlib_list.stdlib_list(
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    return set(std_libs)


def scan_requirements_from_file(filepath: str) -> List[str]:
    """Scans a file for pip requirements specified in comments or docstrings."""
    requirements = []
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()

    # Look for patterns like "# requires: package" or "# pip install package"
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith(("# requires:", "# pip install", "# Requires:", "# Pip install")):
            parts = line.split(
                ":", 1) if ":" in line else line.split("install", 1)
            if len(parts) > 1:
                req = parts[1].strip()
                if req:
                    requirements.append(req)

    return requirements


def main():
    all_imports = set()
    manual_requirements = []

    # Step 1: Collect all imports from all .py files
    for root, _, files in os.walk("."):
        # Skip virtual environments and hidden directories
        if "/venv/" in root or "\\.venv\\" in root or "/\\." in root:
            continue

        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                file_imports = extract_imports_from_file(filepath)
                all_imports.update(file_imports)

                # Also check for manual requirements in comments
                manual_reqs = scan_requirements_from_file(filepath)
                manual_requirements.extend(manual_reqs)

    # Step 2: Check imports against installed packages
    installed_packages = get_installed_packages()
    standard_libs = get_standard_libraries()
    package_mapping = get_package_mapping()

    packages_to_write = []

    # Process normal imports
    for imp in all_imports:
        # Skip standard library modules
        if imp in standard_libs:
            continue

        # Check if this import needs to be mapped to a different package name
        package_name = package_mapping.get(imp, imp).lower()

        # Some modules have dots that need to be converted to dashes
        if "." in package_name:
            # Check if the full module name is in our mapping
            if package_name not in package_mapping.values():
                # Convert dots to dashes for package names
                package_name = package_name.split(".")[0]

        # Skip if it's a standard library
        if package_name in standard_libs:
            continue

        # Try to find the package in installed packages
        if package_name in installed_packages:
            try:
                version = installed_packages[package_name].version
                packages_to_write.append(f"{package_name}=={version}")
            except Exception:
                # If we can't get the version, just add the name
                packages_to_write.append(package_name)

    # Add any manual requirements specified in comments
    for req in manual_requirements:
        # Simple clean-up to standardize format
        req = req.strip().split("#")[0].strip()
        if req and not any(req.lower() in pkg.lower() for pkg in packages_to_write):
            packages_to_write.append(req)

    # Step 3: Add common packages that might be missed due to dynamic imports
    common_packages = [
        "requests",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "torch",
        "transformers",
        "tqdm",
    ]

    for pkg in common_packages:
        if pkg in installed_packages and not any(pkg == p.split("==")[0] for p in packages_to_write):
            try:
                version = installed_packages[pkg].version
                packages_to_write.append(f"{pkg}=={version}")
            except Exception:
                pass

    # Step 4: Write to requirements.txt
    packages_to_write = sorted(set(packages_to_write))
    with open("requirements.txt", "w", encoding="utf-8") as f:
        for pkg in packages_to_write:
            f.write(pkg + "\n")

    print(
        f"✅ Extracted {len(packages_to_write)} packages into requirements.txt"
    )


if __name__ == "__main__":
    main()
