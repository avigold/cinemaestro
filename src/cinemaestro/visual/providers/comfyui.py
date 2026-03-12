"""ComfyUI local provider — executes workflows on a local ComfyUI instance.

This provider supports the full local pipeline: Flux/SDXL image generation
with LoRA and IP-Adapter, AnimateDiff/SVD video generation, and face-swap
post-processing — all through ComfyUI's workflow API.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

import httpx

from cinemaestro.config import ProviderConfig
from cinemaestro.visual.base import GenerationResult, GenerationStatus


class ComfyUIVideoGenerator:
    """Video generation through ComfyUI workflows.

    Uses pre-built workflow templates that are loaded, parameterized, and
    submitted to a running ComfyUI instance.
    """

    provider_name = "comfyui"
    supports_character_reference = True
    supports_image_to_video = True
    max_duration_seconds = 16.0

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.base_url = config.base_url or "http://127.0.0.1:8188"
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    async def generate_video(
        self,
        prompt: str,
        duration_seconds: float = 4.0,
        reference_images: list[Path] | None = None,
        first_frame: Path | None = None,
        last_frame: Path | None = None,
        aspect_ratio: str = "16:9",
        seed: int | None = None,
        output_dir: Path | None = None,
        workflow_path: Path | None = None,
        **kwargs: str,
    ) -> GenerationResult:
        # Load workflow template
        if workflow_path and workflow_path.exists():
            workflow = json.loads(workflow_path.read_text())
        else:
            workflow = self._default_video_workflow()

        # Parameterize workflow
        workflow = self._inject_params(
            workflow,
            prompt=prompt,
            seed=seed,
            first_frame=first_frame,
            reference_images=reference_images,
        )

        # Upload any input images
        if first_frame:
            await self._upload_image(first_frame)
        if reference_images:
            for img in reference_images:
                await self._upload_image(img)

        # Submit workflow
        client_id = str(uuid.uuid4())
        resp = await self.client.post(
            "/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]

        # Poll for completion
        result = await self._poll_completion(prompt_id, output_dir)
        return result

    async def _upload_image(self, image_path: Path) -> str:
        """Upload an image to ComfyUI's input directory."""
        files = {
            "image": (image_path.name, image_path.read_bytes(), "image/png"),
        }
        resp = await self.client.post("/upload/image", files=files)
        resp.raise_for_status()
        return resp.json().get("name", image_path.name)

    async def _poll_completion(
        self, prompt_id: str, output_dir: Path | None, poll_interval: float = 2.0
    ) -> GenerationResult:
        while True:
            resp = await self.client.get(f"/history/{prompt_id}")
            resp.raise_for_status()
            history = resp.json()

            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})

                if status.get("completed", False):
                    outputs = entry.get("outputs", {})
                    output_path = None

                    # Find the video output
                    for node_id, node_output in outputs.items():
                        gifs = node_output.get("gifs", [])
                        if gifs:
                            filename = gifs[0]["filename"]
                            subfolder = gifs[0].get("subfolder", "")
                            if output_dir:
                                output_path = output_dir / filename
                                video_resp = await self.client.get(
                                    "/view",
                                    params={
                                        "filename": filename,
                                        "subfolder": subfolder,
                                        "type": "output",
                                    },
                                )
                                video_resp.raise_for_status()
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                output_path.write_bytes(video_resp.content)
                            break

                    return GenerationResult(
                        provider=self.provider_name,
                        generation_id=prompt_id,
                        status=GenerationStatus.COMPLETED,
                        output_path=output_path,
                    )

                if status.get("status_str") == "error":
                    return GenerationResult(
                        provider=self.provider_name,
                        generation_id=prompt_id,
                        status=GenerationStatus.FAILED,
                        error_message=str(status.get("messages", "Unknown error")),
                    )

            await asyncio.sleep(poll_interval)

    def _inject_params(
        self,
        workflow: dict,
        prompt: str,
        seed: int | None = None,
        first_frame: Path | None = None,
        reference_images: list[Path] | None = None,
    ) -> dict:
        """Inject parameters into workflow nodes.

        Looks for known node titles and updates their inputs.
        """
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            class_type = node.get("class_type", "")

            # Update text prompts
            if class_type in ("CLIPTextEncode", "CLIPTextEncodeSDXL"):
                if "text" in inputs:
                    inputs["text"] = prompt

            # Update seeds
            if seed is not None and "seed" in inputs:
                inputs["seed"] = seed
            if seed is not None and "noise_seed" in inputs:
                inputs["noise_seed"] = seed

            # Update input images
            if first_frame and class_type == "LoadImage":
                inputs["image"] = first_frame.name

        return workflow

    def _default_video_workflow(self) -> dict:
        """Minimal default workflow — users should provide custom workflows."""
        return {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["2", 0]},
            },
        }

    async def check_status(self, generation_id: str) -> GenerationStatus:
        resp = await self.client.get(f"/history/{generation_id}")
        resp.raise_for_status()
        history = resp.json()
        if generation_id in history:
            status = history[generation_id].get("status", {})
            if status.get("completed"):
                return GenerationStatus.COMPLETED
            if status.get("status_str") == "error":
                return GenerationStatus.FAILED
            return GenerationStatus.PROCESSING
        return GenerationStatus.PENDING


class ComfyUIImageGenerator:
    """Image generation through ComfyUI workflows."""

    provider_name = "comfyui"
    supports_lora = True
    supports_ip_adapter = True

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.base_url = config.base_url or "http://127.0.0.1:8188"
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.config.timeout_seconds,
            )
        return self._client

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: str = "",
        reference_images: list[Path] | None = None,
        lora_path: Path | None = None,
        lora_trigger_word: str = "",
        width: int = 1920,
        height: int = 1080,
        seed: int | None = None,
        output_dir: Path | None = None,
        workflow_path: Path | None = None,
        **kwargs: str,
    ) -> GenerationResult:
        if lora_trigger_word and lora_trigger_word not in prompt:
            prompt = f"{lora_trigger_word}, {prompt}"

        if workflow_path and workflow_path.exists():
            workflow = json.loads(workflow_path.read_text())
        else:
            workflow = self._default_image_workflow()

        # Parameterize
        for node_id, node in workflow.items():
            inputs = node.get("inputs", {})
            class_type = node.get("class_type", "")

            if class_type in ("CLIPTextEncode",) and "text" in inputs:
                inputs["text"] = prompt
            if seed is not None and "seed" in inputs:
                inputs["seed"] = seed
            if class_type == "EmptyLatentImage":
                inputs["width"] = width
                inputs["height"] = height

        if reference_images:
            for img in reference_images:
                await self._upload_image(img)

        client_id = str(uuid.uuid4())
        resp = await self.client.post(
            "/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        resp.raise_for_status()
        prompt_id = resp.json()["prompt_id"]

        return await self._poll_completion(prompt_id, output_dir)

    async def _upload_image(self, image_path: Path) -> str:
        files = {
            "image": (image_path.name, image_path.read_bytes(), "image/png"),
        }
        resp = await self.client.post("/upload/image", files=files)
        resp.raise_for_status()
        return resp.json().get("name", image_path.name)

    async def _poll_completion(
        self, prompt_id: str, output_dir: Path | None, poll_interval: float = 2.0
    ) -> GenerationResult:
        while True:
            resp = await self.client.get(f"/history/{prompt_id}")
            resp.raise_for_status()
            history = resp.json()

            if prompt_id in history:
                entry = history[prompt_id]
                status = entry.get("status", {})

                if status.get("completed", False):
                    outputs = entry.get("outputs", {})
                    output_path = None

                    for node_id, node_output in outputs.items():
                        images = node_output.get("images", [])
                        if images:
                            filename = images[0]["filename"]
                            subfolder = images[0].get("subfolder", "")
                            if output_dir:
                                output_path = output_dir / filename
                                img_resp = await self.client.get(
                                    "/view",
                                    params={
                                        "filename": filename,
                                        "subfolder": subfolder,
                                        "type": "output",
                                    },
                                )
                                img_resp.raise_for_status()
                                output_path.parent.mkdir(parents=True, exist_ok=True)
                                output_path.write_bytes(img_resp.content)
                            break

                    return GenerationResult(
                        provider=self.provider_name,
                        generation_id=prompt_id,
                        status=GenerationStatus.COMPLETED,
                        output_path=output_path,
                    )

                if status.get("status_str") == "error":
                    return GenerationResult(
                        provider=self.provider_name,
                        generation_id=prompt_id,
                        status=GenerationStatus.FAILED,
                        error_message=str(status.get("messages", "")),
                    )

            await asyncio.sleep(poll_interval)

    def _default_image_workflow(self) -> dict:
        return {
            "1": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["2", 0]},
            },
        }
