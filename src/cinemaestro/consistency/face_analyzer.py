"""Face analysis using InsightFace for character consistency verification.

Extracts face embeddings from video frames and compares them against
stored character reference embeddings to detect identity drift.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cinemaestro.consistency.base import ConsistencyScore
from cinemaestro.config import ConsistencyConfig

logger = logging.getLogger(__name__)


class FaceAnalyzer:
    """Analyzes face identity consistency in generated video clips."""

    def __init__(self, config: ConsistencyConfig) -> None:
        self.config = config
        self._app = None

    def _load_app(self):  # type: ignore[no-untyped-def]
        if self._app is None:
            import insightface

            self._app = insightface.app.FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=0, det_size=(640, 640))
        return self._app

    def extract_embedding(self, image_path: Path) -> np.ndarray | None:
        """Extract face embedding from a single image."""
        import cv2

        app = self._load_app()
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        faces = app.get(img)
        if not faces:
            return None

        # Return embedding of the largest face
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return largest.embedding

    def save_embedding(self, embedding: np.ndarray, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), embedding)

    def load_embedding(self, embedding_path: Path) -> np.ndarray:
        return np.load(str(embedding_path))

    async def check_shot(
        self,
        video_path: Path,
        character_id: str,
        reference_embedding_path: Path,
        threshold: float | None = None,
        sample_interval: float | None = None,
    ) -> ConsistencyScore:
        """Check face consistency across frames of a video clip."""
        import asyncio
        import cv2

        if threshold is None:
            threshold = self.config.face_similarity_threshold
        if sample_interval is None:
            sample_interval = self.config.sample_interval_seconds

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._check_shot_sync,
            video_path,
            character_id,
            reference_embedding_path,
            threshold,
            sample_interval,
        )

    def _check_shot_sync(
        self,
        video_path: Path,
        character_id: str,
        reference_embedding_path: Path,
        threshold: float,
        sample_interval: float,
    ) -> ConsistencyScore:
        import cv2

        app = self._load_app()
        ref_embedding = self.load_embedding(reference_embedding_path)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        frame_interval = int(fps * sample_interval)

        similarities: list[float] = []
        failing_frames: list[int] = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                faces = app.get(frame)
                if faces:
                    # Find the face most similar to reference
                    best_sim = 0.0
                    for face in faces:
                        sim = self._cosine_similarity(ref_embedding, face.embedding)
                        best_sim = max(best_sim, sim)

                    similarities.append(best_sim)
                    if best_sim < threshold:
                        failing_frames.append(frame_idx)

            frame_idx += 1

        cap.release()

        if not similarities:
            return ConsistencyScore(
                shot_id="",
                character_id=character_id,
                mean_similarity=0.0,
                min_similarity=0.0,
                max_similarity=0.0,
                frames_checked=0,
                frames_below_threshold=0,
                passed=True,  # no faces found, nothing to fail
            )

        return ConsistencyScore(
            shot_id="",
            character_id=character_id,
            mean_similarity=float(np.mean(similarities)),
            min_similarity=float(np.min(similarities)),
            max_similarity=float(np.max(similarities)),
            frames_checked=len(similarities),
            frames_below_threshold=len(failing_frames),
            passed=len(failing_frames) == 0,
            failing_frame_indices=failing_frames,
        )

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)
