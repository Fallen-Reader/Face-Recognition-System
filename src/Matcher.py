import numpy as np
from config import MATCH_TOLERANCE


class FaceMatcher:
    def match(self, unknown_encoding, known_encodings, known_names, tolerance: float = MATCH_TOLERANCE):
        
        if len(known_encodings) == 0:
            return "Unknown", None

        distances = np.linalg.norm(np.asarray(known_encodings) - unknown_encoding, axis=1)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])

        if best_distance <= tolerance:
            return known_names[best_idx], best_distance
        return "Unknown", best_distance