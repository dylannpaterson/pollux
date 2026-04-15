from abc import ABC, abstractmethod
import numpy as np

class WCSAdapter:
    """
    Adapter to provide a consistent interface for both astropy.wcs.WCS 
    and galsim.WCS (GSFitsWCS) objects.
    """
    def __init__(self, wcs_obj):
        self.wcs = wcs_obj
        # Galsim-style objects have xyToradec but not pixel_to_world_values
        # IMPORTANT: romanisim.wcs.GWCS also has xyToradec (to mimic Galsim) 
        # but it follows 0-based indexing like GWCS.
        # We check for 'galsim' in the module name to be sure.
        self.is_galsim = (hasattr(wcs_obj, 'xyToradec') and 
                          not hasattr(wcs_obj, 'pixel_to_world_values') and
                          'galsim' in str(type(wcs_obj)).lower())

    def pixel_to_world_values(self, x, y):
        if self.is_galsim:
            import galsim
            # Galsim is 1-based by default for xyToradec
            ra, dec = self.wcs.xyToradec(x + 1, y + 1, units=galsim.degrees)
            return ra, dec
        elif hasattr(self.wcs, 'pixel_to_world_values'):
            return self.wcs.pixel_to_world_values(x, y)
        elif hasattr(self.wcs, 'xyToradec'):
            # This handles romanisim.wcs.GWCS which mimics Galsim but is 0-based
            # or it handles Galsim if we missed it above but want to try 0-based
            return self.wcs.xyToradec(x, y)
        elif callable(self.wcs):
            # GWCS or similar callable object
            return self.wcs(x, y)
        else:
            raise AttributeError(f"WCS object {type(self.wcs)} has no known pixel-to-world method.")

    def world_to_pixel_values(self, ra, dec):
        if self.is_galsim:
            import galsim
            x, y = self.wcs.radecToxy(ra, dec, units=galsim.degrees)
            return x - 1, y - 1
        elif hasattr(self.wcs, 'world_to_pixel_values'):
            return self.wcs.world_to_pixel_values(ra, dec)
        elif hasattr(self.wcs, 'radecToxy'):
            # This handles romanisim.wcs.GWCS which mimics Galsim but is 0-based
            return self.wcs.radecToxy(ra, dec)
        elif hasattr(self.wcs, 'backward_transform'):
            return self.wcs.backward_transform(ra, dec)
        else:
            raise AttributeError(f"WCS object {type(self.wcs)} has no known world-to-pixel method.")

class PipelineContext:
    """Holds the state of the pipeline between steps."""
    def __init__(self):
        self.image_data = None
        self.wcs = None # Will be wrapped in WCSAdapter
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
