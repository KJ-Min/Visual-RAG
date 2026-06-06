from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Protocol

import numpy as np
from PIL import Image


@dataclass
class OCRResult:
    text: str
    metadata: dict


class OCREngine(Protocol):
    name: str

    def extract(self, image_path: str | Path) -> OCRResult:
        ...


class PytesseractEngine:
    name = "pytesseract"

    def __init__(self, lang: str | None = None, config: str | None = None):
        try:
            import pytesseract
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "pytesseract is not installed. Install it with: pip install pytesseract"
            ) from exc
        if shutil.which("tesseract") is None:
            raise RuntimeError(
                "The tesseract binary is not on PATH. Install it with: conda install -c conda-forge tesseract -y"
            )

        self._pytesseract = pytesseract
        self.lang = lang
        self.config = config

    def extract(self, image_path: str | Path) -> OCRResult:
        image = Image.open(image_path).convert("RGB")
        kwargs = {}
        if self.lang:
            kwargs["lang"] = self.lang
        if self.config:
            kwargs["config"] = self.config
        text = self._pytesseract.image_to_string(image, **kwargs)
        return OCRResult(text=text, metadata={"engine": self.name, "lang": self.lang})


class PPOCRv3Engine:
    """Adapter for the official FastDeploy PP-OCRv3 demo logic."""

    def __init__(
        self,
        *,
        mode: str,
        det_model_dir: str | Path,
        rec_model_dir: str | Path,
        cls_model_dir: str | Path,
        rec_label_file: str | Path,
        backend: str = "gpu",
        device_id: int = 0,
        min_score: float = 0.6,
    ):
        if mode not in {"layout_preserving", "adjacent_merging"}:
            raise ValueError("mode must be layout_preserving or adjacent_merging")
        import cv2
        import fastdeploy as fd

        self.name = mode
        self._cv2 = cv2
        self._fd = fd
        self.min_score = min_score
        self.backend = backend
        self.device_id = device_id

        det_option, cls_option, rec_option = self._build_option()
        det_model_dir = Path(det_model_dir)
        rec_model_dir = Path(rec_model_dir)
        cls_model_dir = Path(cls_model_dir)

        det_model = fd.vision.ocr.DBDetector(
            str(det_model_dir / "inference.pdmodel"),
            str(det_model_dir / "inference.pdiparams"),
            runtime_option=det_option,
        )
        cls_model = fd.vision.ocr.Classifier(
            str(cls_model_dir / "inference.pdmodel"),
            str(cls_model_dir / "inference.pdiparams"),
            runtime_option=cls_option,
        )
        rec_model = fd.vision.ocr.Recognizer(
            str(rec_model_dir / "inference.pdmodel"),
            str(rec_model_dir / "inference.pdiparams"),
            str(rec_label_file),
            runtime_option=rec_option,
        )

        det_model.preprocessor.max_side_len = 960
        det_model.postprocessor.det_db_thresh = 0.3
        det_model.postprocessor.det_db_box_thresh = 0.6
        det_model.postprocessor.det_db_unclip_ratio = 1.5
        det_model.postprocessor.det_db_score_mode = "slow"
        det_model.postprocessor.use_dilation = False
        cls_model.postprocessor.cls_thresh = 0.9

        self._model = fd.vision.ocr.PPOCRv3(
            det_model=det_model,
            cls_model=cls_model,
            rec_model=rec_model,
        )

    def _build_option(self):
        fd = self._fd
        det_option = fd.RuntimeOption()
        cls_option = fd.RuntimeOption()
        rec_option = fd.RuntimeOption()
        if self.backend.lower() == "gpu":
            det_option.use_gpu(self.device_id)
            cls_option.use_gpu(self.device_id)
            rec_option.use_gpu(self.device_id)
        else:
            det_option.use_cpu()
            cls_option.use_cpu()
            rec_option.use_cpu()
        return det_option, cls_option, rec_option

    def extract(self, image_path: str | Path) -> OCRResult:
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)
        image = self._cv2.cvtColor(image, self._cv2.COLOR_RGB2BGR)
        result = self._model.predict(image)
        text = self._to_layout_text(result)
        return OCRResult(
            text=text,
            metadata={
                "engine": self.name,
                "backend": self.backend,
                "min_score": self.min_score,
                "boxes": len(getattr(result, "boxes", [])),
            },
        )

    def _to_layout_text(self, result) -> str:
        text_boxes = []
        for box, text, score in zip(result.boxes, result.text, result.rec_scores):
            if score < self.min_score:
                continue
            coords = [(box[i], box[i + 1]) for i in range(0, len(box), 2)]
            center_x = (coords[0][0] + coords[2][0]) / 2
            center_y = (coords[0][1] + coords[2][1]) / 2
            text_boxes.append((center_x, center_y, text))

        text_boxes.sort(key=lambda item: (item[1], item[0]))
        merged = []
        previous = None
        for current in text_boxes:
            if previous is not None:
                spaces, newlines = _spaces_and_newlines(current, previous)
                merged.append("\n" * newlines + " " * spaces)
            merged.append(current[2])
            previous = current
        return "".join(merged)


def _spaces_and_newlines(current_box, previous_box, space_threshold=45, line_threshold=15):
    if abs(current_box[1] - previous_box[1]) < line_threshold:
        return max(1, int(abs(current_box[0] - previous_box[0]) / space_threshold)), 0
    return 0, max(1, int(abs(current_box[1] - previous_box[1]) / line_threshold))


def get_ocr_engine(engine: str, **kwargs) -> OCREngine:
    if engine == "pytesseract":
        return PytesseractEngine(lang=kwargs.get("lang"), config=kwargs.get("tesseract_config"))
    if engine in {"layout_preserving", "adjacent_merging"}:
        required = ["det_model_dir", "rec_model_dir", "cls_model_dir", "rec_label_file"]
        missing = [name for name in required if not kwargs.get(name)]
        if missing:
            raise ValueError(f"{engine} requires: {', '.join(missing)}")
        return PPOCRv3Engine(mode=engine, **{k: v for k, v in kwargs.items() if v is not None})
    raise ValueError("engine must be pytesseract, layout_preserving, or adjacent_merging")
