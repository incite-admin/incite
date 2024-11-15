# Concolic Testing for CNNs

This project implements concolic testing on a convolutional neural network (CNN) using SHAP values to analyze and explore critical decision points in neural networks.

## Prerequisites

- [Python](https://www.python.org/downloads/) version == 3.8.5<br>
  Basically, it should also work for other versions not lower than 3.8. Simply follow the usual installation instructions for Python.<br>

- [CVC4](https://github.com/CVC4/CVC4) commit version == [d1f3225e26b9d64f065048885053392b10994e715](https://github.com/cvc5/cvc5/blob/d1f3225e26b9d64f065048885053392b10994e71/INSTALL.md)<br>
  Since our CVC4 version has to cope with that of the base project PyExZ3 when we compare the performance of the two, our designated version above cannot be the latest. Otherwise, the CVC4 Python API bindings used in PyExZ3 cannot work.<br>The installation instructions for CVC4 is almost the same as that in the provided link, except that the configuration command should be modified to `./configure.sh --language-bindings=python --python3` for the use of CVC4 Python API bindings. A user must ensure by himself/herself that the command `cvc4` can be found by an operating system shell. Otherwise the tool may not work.<br>

- [pipenv](https://pypi.org/project/pipenv/)<br>
  This is required for the use of the virtual environment mechanism in our project. Install it as a usual Python package.<br>

- additional settings<br>
  1. For CVC4 to be findable by the Python API, `export PYTHONPATH={path-to-CVC4-build-folder}/src/bindings/python` should be put in `~/.bashrc`.
  2. For pipenv to create a virtual environment in each project folder, `export PIPENV_VENV_IN_PROJECT=1` should be put in `~/.bashrc`, too.


## Setup Instructions

1. **Calculate SHAP Values:**

   You need to precompute the SHAP values for the first layer of your model. Ensure that these SHAP values are saved in a newly created directory named `shap_value`.

2. **Create the `shap_value` Directory:**

   ```
   mkdir shap_value
   ```

   Place the calculated SHAP values for the first layer in this directory.

## Running the Attack

To execute the attack on a CNN model, use the following command:

```bash
python3 test_dnnct_cnn.py
```

## Notes

- Ensure that all dependencies are installed and your Python environment is correctly set up.
- Precomputed SHAP values for the first layer must be present in the `shap_value` directory before running the script, as they are essential for the analysis.

## Troubleshooting

If you encounter any issues, please check the following:

- Ensure the SHAP values are correctly formatted and accessible in the `shap_value` directory.
- Verify that all necessary Python libraries are installed.

For further assistance, please refer to the project documentation or contact the project maintainers.
