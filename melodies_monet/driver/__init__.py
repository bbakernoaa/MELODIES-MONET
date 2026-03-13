from melodies_monet.driver.data import Data
from melodies_monet.driver.pair import pair
from melodies_monet.driver.analysis import analysis

# For backward compatibility
model = Data
observation = Data

__all__ = (
    "pair",
    "Data",
    "analysis",
)
