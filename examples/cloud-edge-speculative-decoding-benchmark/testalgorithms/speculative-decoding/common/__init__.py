"""Common utilities shared by speculative decoding algorithms.

Algorithm modules should primarily import from the short public modules:

- `common.config`
- `common.decorators`
- `common.modeling`
- `common.payload`
- `common.schema`
- `common.timing`

`common.runtime` is used by the shared base classes to handle request
normalization, session state, network simulation, response formatting, and
sample-output logging.
"""
