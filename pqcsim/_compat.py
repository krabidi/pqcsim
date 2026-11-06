"""Small numpy version shims (pqcsim supports numpy 1.21+ and 2.x)."""
import numpy as np

trapezoid = getattr(np, "trapezoid", None) or np.trapz
