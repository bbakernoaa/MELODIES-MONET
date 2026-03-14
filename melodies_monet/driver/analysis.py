from melodies_monet.driver.orchestrator import orchestrator


class analysis(orchestrator):
    """
    Backward compatibility class for the legacy 'analysis' name.
    Now inherits from and redirects to the 'orchestrator' class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def open_models(self, **kwargs):
        """Legacy method for opening models. Redirects to open_data."""
        return self.open_data(**kwargs)

    def open_obs(self, **kwargs):
        """Legacy method for opening observations. Redirects to open_data."""
        return self.open_data(**kwargs)
