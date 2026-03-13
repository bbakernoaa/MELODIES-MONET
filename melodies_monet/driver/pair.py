class pair:
    """The pair class.

    The pair class pairs model data
    directly with reference (observational or model) data along time and space.
    """

    def __init__(self):
        """Initialize a :class:`pair` object."""
        self.type = "pt_sfc"
        self.radius_of_influence = 1e6
        self.ref = None
        self.model = None
        self.model_vars = None
        self.ref_vars = None
        self.filename = None

    def __repr__(self):
        return (
            f"{type(self).__name__}(\n"
            f"    type={self.type!r},\n"
            f"    radius_of_influence={self.radius_of_influence!r},\n"
            f"    ref={self.ref!r},\n"
            f"    model={self.model!r},\n"
            f"    model_vars={self.model_vars!r},\n"
            f"    ref_vars={self.ref_vars!r},\n"
            f"    filename={self.filename!r},\n"
            ")"
        )
