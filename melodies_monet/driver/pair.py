# SPDX-License-Identifier: Apache-2.0
#
class pair:
    """The pair class.

    The pair class pairs model data
    directly with reference (observational or model) data along time and space.
    """

    def __init__(self):
        """Initialize a :class:`pair` object."""
        self.type = "pt_sfc"
        self.ref = None
        self.model = None
        self.model_vars = None
        self.ref_vars = None
        self.filename = None
        self.model_plot_kwargs = {}
        self.ref_plot_kwargs = {}

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"    type={self.type!r},\n"
            f"    ref={self.ref!r},\n"
            f"    model={self.model!r},\n"
            f"    model_vars={self.model_vars!r},\n"
            f"    ref_vars={self.ref_vars!r},\n"
            f"    filename={self.filename!r},\n"
            f"    model_plot_kwargs={self.model_plot_kwargs!r},\n"
            f"    ref_plot_kwargs={self.ref_plot_kwargs!r},\n"
            ")"
        )
