Install Python Bindings
------------------------

To setup the Python bindings once you've cloned _elsa_, change into the cloned directory and simply run
`pip install .`. This will build _elsa_ and install it, including the Python bindings. In case you
wonder what is happening, add the `--verbose` flag to see the progress.

### Verify installation

Once everything is set up, simply open a Python interpreter and run
```python
import pyelsa as elsa
```
to check if everything is working.

### Install with debug information

In order to generate debug information for the Python bindings, run
`env CMAKE_ARGS="-DCMAKE_BUILD_TYPE:STRING=Debug" pip install .`
