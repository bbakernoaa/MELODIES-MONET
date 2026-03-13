from melodies_monet.driver.orchestrator import orchestrator

class analysis(orchestrator):
    """
    Backward compatibility class for the legacy 'analysis' name.
    Now inherits from and redirects to the 'orchestrator' class.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
