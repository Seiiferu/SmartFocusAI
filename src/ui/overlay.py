import cv2
from typing import List

class OverlayDrawer:
    """
    Dessine en overlay sur la frame :
     - direction du regard
     - compteur de clignements
     - objets distracteurs détectés
     - état de parole
     - statut final (concentré/distrait)
    """
    def __init__(self,
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 font_scale=0.6,
                 thickness=2):
        self.font       = font
        self.font_scale = font_scale
        self.thickness  = thickness
        # Couleurs BGR
        self.color_text     = (255, 255, 255)  # blanc
        self.color_bg       = (0, 0, 0)        # arrière‑plan noir
        self.color_conc     = (0, 200, 0)      # vert pour concentré
        self.color_dist     = (0, 0, 200)      # rouge pour distrait

    def draw(self,
             frame,
             gaze_dir: str,
             blink_count: int,
             distractors: List[str],
             speaking: bool,
             state: str):
        h, w = frame.shape[:2]
        lines = [
            f"Regard: {gaze_dir}",
            f"Clignements: {blink_count}",
            f"Distracteurs: {','.join(distractors) or 'aucun'}",
            f"Parle: {speaking}",
            f"Statut: {state}"
        ]

        # Dessiner un petit panneau noir semi‑transparent en haut à gauche
        overlay = frame.copy()
        cv2.rectangle(overlay, (0,0), (300, 20*len(lines)+10), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Écrire chaque ligne
        for i, text in enumerate(lines):
            y = 20 + i*20
            # Choix de la couleur pour la dernière ligne
            color = self.color_conc if state=="concentré" else self.color_dist
            if i < len(lines)-1:
                color = self.color_text
            cv2.putText(frame, text, (10, y),
                        self.font, self.font_scale, color, self.thickness)
