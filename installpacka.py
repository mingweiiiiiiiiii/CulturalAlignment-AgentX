import os
import ast
import sys
import pkg_resources
import stdlib_list

def extract_imports_from_file(filepath):
    """Extracts imported module names from a single Python file."""
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
                imports.add(alias.name.split('.')[0])
        elif isinstance(item, ast.ImportFrom):
            if item.module:
                imports.add(item.module.split('.')[0])
    return imports

def get_installed_packages():
    """Gets a set of installed package names."""
    return {pkg.key for pkg in pkg_resources.working_set}

def get_standard_libraries():
    """Gets a set of standard libraries based on Python version."""
    std_libs = stdlib_list.stdlib_list(f"{sys.version_info.major}.{sys.version_info.minor}")
    return set(std_libs)

def main():
    all_imports = set()

    # Step 1: Collect all imports from all .py files
    for root, _, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                all_imports.update(extract_imports_from_file(filepath))

    installed_packages = get_installed_packages()
    standard_libs = get_standard_libraries()

    # Step 2: Filter only real installed 3rd-party packages
    packages_to_write = []
    for imp in all_imports:
        if imp.lower() in installed_packages and imp not in standard_libs:
            try:
                version = pkg_resources.get_distribution(imp).version
                packages_to_write.append(f"{imp}=={version}")
            except Exception:
                # Ignore if unable to find version
                pass

    # Step 3: Write to requirements.txt
    packages_to_write = sorted(set(packages_to_write))
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        for pkg in packages_to_write:
            f.write(pkg + '\n')

    print(f"✅ Extracted {len(packages_to_write)} third-party packages into requirements.txt")

if __name__ == "__main__":
    main()
