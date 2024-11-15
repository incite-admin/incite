# SHAP-based Concolic Testing for Transformers

This project implements concolic testing on a Transformer model, utilizing SHAP values as a priority queue influence matrix to enhance its robustness and performance evaluation.

## Prerequisites

Before running the project, ensure you have the following prerequisites installed:

- **[Python](https://www.python.org/downloads/):** Version 3.8.5  
  The project should work with any version not lower than 3.8. Please follow the usual Python installation instructions.

- **[CVC4](https://github.com/CVC4/CVC4):** Commit version [d1f3225e26b9d64f065048885053392b10994e71](https://github.com/cvc5/cvc5/blob/d1f3225e26b9d64f065048885053392b10994e71/INSTALL.md)  
  The specific version is required to maintain compatibility with the base project PyExZ3. Newer versions might cause incompatibility issues with the CVC4 Python API bindings used in PyExZ3. Follow the installation instructions provided in the link, but modify the configuration command to `./configure.sh --language-bindings=python --python3` to enable CVC4 Python API bindings. Ensure the `cvc4` command is available in your shell's PATH.

- **[pipenv](https://pypi.org/project/pipenv/):**  
  Required to manage the virtual environment. Install it using pip.

- **Additional settings:**  
  1. To ensure CVC4 is accessible by the Python API, add `export PYTHONPATH={path-to-CVC4-build-folder}/src/bindings/python` to your `~/.bashrc`.
  2. To create a virtual environment in each project folder with pipenv, add `export PIPENV_VENV_IN_PROJECT=1` to your `~/.bashrc`.

## Running the Attack

To execute the attack on a Transformer model using SHAP values as a priority queue influence matrix, use the following command in your terminal:

```bash
python3 dnnct_transformer_multi.py
```

## Notes

- Ensure all dependencies are installed and your Python environment is correctly configured.
- Precomputed SHAP values for the first layer must be available in the `shap_value` directory before running the script, as they are essential for prioritizing the influence of different inputs.

