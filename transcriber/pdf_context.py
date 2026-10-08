"""발표자료 PDF를 모델 입력용 문맥(텍스트 + 페이지 이미지)으로 변환한다.

PDF는 요약하지 않는다. 슬라이드에 적힌 용어/약어/수식을 그대로 모델에 보여 주어
음성 인식이 불확실한 부분을 보정하는 근거로만 쓴다.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# 이보다 텍스트가 적은 페이지는 그림/스캔 슬라이드로 보고 이미지로 전달한다.
MIN_TEXT_CHARS = 40


@dataclass
class PdfContext:
    text: str = ""
    images: list = field(default_factory=list)  # PIL.Image
    sources: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.text and not self.images


def build_pdf_context(pdf_paths: list[Path], cfg: dict) -> PdfContext:
    mode = str(cfg.get("mode", "auto")).lower()
    ctx = PdfContext()
    if mode == "off" or not pdf_paths:
        return ctx

    import pymupdf
    from PIL import Image

    max_chars = int(cfg.get("max_text_chars", 12000))
    max_images = int(cfg.get("max_images", 8))
    dpi = int(cfg.get("image_dpi", 100))

    text_parts: list[str] = []
    image_pages = []  # (doc, page_index)
    docs = []
    for path in pdf_paths:
        doc = pymupdf.open(path)
        docs.append(doc)
        ctx.sources.append(path.name)
        for i, page in enumerate(doc):
            page_text = " ".join(page.get_text("text").split())
            has_text = len(page_text) >= MIN_TEXT_CHARS
            if mode in ("auto", "text") and page_text:
                text_parts.append(f"[{path.name} p.{i + 1}] {page_text}")
            if mode == "images" or (mode == "auto" and not has_text):
                image_pages.append((doc, i))

    text = "\n".join(text_parts)
    if len(text) > max_chars:
        log.info("PDF 텍스트가 길어 앞부분 %d자만 사용합니다 (전체 %d자).", max_chars, len(text))
        text = text[:max_chars]
    ctx.text = text

    if len(image_pages) > max_images:
        log.info("이미지로 전달할 페이지가 %d개라 %d개만 사용합니다.", len(image_pages), max_images)
        # 앞쪽에 몰리지 않게 고르게 선택
        stride = len(image_pages) / max_images
        image_pages = [image_pages[int(k * stride)] for k in range(max_images)]
    for doc, i in image_pages:
        pix = doc[i].get_pixmap(dpi=dpi)
        ctx.images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))

    for doc in docs:
        doc.close()
    return ctx
