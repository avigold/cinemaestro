"""Face swap correction using ReActor/inswapper for identity repair.

When the consistency checker detects identity drift in generated video frames,
this module applies targeted face replacement using a reference image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class FaceSwapper:
    """Applies face correction to video frames where identity has drifted."""

    def __init__(self, model_path: str = "inswapper_128.onnx") -> None:
        self.model_path = model_path
        self._swapper = None
        self._face_app = None

    def _load(self) -> None:
        if self._swapper is not None:
            return

        import insightface

        self._face_app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._face_app.prepare(ctx_id=0, det_size=(640, 640))

        self._swapper = insightface.model_zoo.get_model(
            self.model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    async def repair_shot(
        self,
        video_path: Path,
        character_id: str,
        reference_image: Path,
        failing_frames: list[int],
        output_path: Path,
    ) -> Path:
        """Replace faces in failing frames with the reference character face."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._repair_sync,
            video_path,
            reference_image,
            failing_frames,
            output_path,
        )

    def _repair_sync(
        self,
        video_path: Path,
        reference_image: Path,
        failing_frames: list[int],
        output_path: Path,
    ) -> Path:
        import cv2

        self._load()
        assert self._face_app is not None
        assert self._swapper is not None

        # Get source face from reference image
        src_img = cv2.imread(str(reference_image))
        src_faces = self._face_app.get(src_img)
        if not src_faces:
            logger.error("No face found in reference image: %s", reference_image)
            return video_path
        source_face = max(
            src_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )

        # Process video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        failing_set = set(failing_frames)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in failing_set:
                # Swap face in this frame
                dst_faces = self._face_app.get(frame)
                if dst_faces:
                    target_face = max(
                        dst_faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    )
                    frame = self._swapper.get(frame, target_face, source_face, paste_back=True)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        logger.info(
            "Repaired %d frames in %s -> %s",
            len(failing_frames),
            video_path,
            output_path,
        )
        return output_path
