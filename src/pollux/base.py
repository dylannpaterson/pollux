from abc import ABC, abstractmethod

class PipelineContext:
    """Holds the state of the pipeline between steps."""
    def __init__(self):
        self.image_data = None
        self.wcs = None
        self.filter_name = None
        self.catalog = None
        self.metadata = {}

class PipelineStep(ABC):
    """Base class for all pipeline steps."""
    @abstractmethod
    def run(self, context: PipelineContext, config: dict):
        pass

    def finalize(self, config: dict):
        """Optional clean-up or final bulk operations."""
        pass
