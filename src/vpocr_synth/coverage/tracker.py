"""Coverage tracking for synthetic plate datasets."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateText, PlateType

WIDTH_PX_BINS: list[str] = ["<40px", "40-60px", "60-90px", "90-140px", ">=140px"]
CHAR_HEIGHT_PX_BINS: list[str] = ["<8px", "8-12px", "12-18px", "18-28px", ">=28px"]
YAW_BINS: list[str] = ["front_0_20", "oblique_20_45", "steep_45_70", "side_70_plus"]
FIELDS: list[str] = ["region", "classification", "hiragana", "serial"]


def width_px_bin(width: float) -> str:
    if width < 40:
        return "<40px"
    if width < 60:
        return "40-60px"
    if width < 90:
        return "60-90px"
    if width < 140:
        return "90-140px"
    return ">=140px"


def char_height_px_bin(h: float) -> str:
    if h < 8:
        return "<8px"
    if h < 12:
        return "8-12px"
    if h < 18:
        return "12-18px"
    if h < 28:
        return "18-28px"
    return ">=28px"


def yaw_bin(yaw_deg: float) -> str:
    a = abs(yaw_deg)
    if a < 20:
        return "front_0_20"
    if a < 45:
        return "oblique_20_45"
    if a < 70:
        return "steep_45_70"
    return "side_70_plus"


class CoverageTracker:
    """Records samples and reports 2D/3D coverage cross-tabs and blind spots."""

    WIDTH_BINS = WIDTH_PX_BINS
    CHAR_HEIGHT_BINS = CHAR_HEIGHT_PX_BINS
    YAW_BINS = YAW_BINS
    FIELDS = FIELDS
    # aliases
    width_px_bins = WIDTH_PX_BINS
    char_height_px_bins = CHAR_HEIGHT_PX_BINS
    yaw_bins = YAW_BINS

    def __init__(self) -> None:
        self.regions = PlateDataLoader.load_regions()
        self.region_names = [r.name for r in self.regions]
        self.charset = PlateDataLoader.load_charset()
        self.total_samples = 0
        self.region_counts: Counter[str] = Counter({n: 0 for n in self.region_names})
        self.char_counts: Counter[str] = Counter({c: 0 for c in self.charset})
        self.plate_type_counts: Counter[str] = Counter()
        # 3D matrices as Counter with tuple keys
        self._matrix_iwy: Counter[tuple[str, str, str]] = Counter()  # illumination, width, yaw
        self._matrix_fcd: Counter[tuple[str, str, str]] = Counter()  # field, char_height, degradation
        self._illuminations: set[str] = set()
        self._degradations: set[str] = set()

    # bin helpers as methods
    @staticmethod
    def width_px_bin(width: float) -> str:
        return width_px_bin(width)

    @staticmethod
    def char_height_px_bin(h: float) -> str:
        return char_height_px_bin(h)

    @staticmethod
    def yaw_bin(yaw_deg: float) -> str:
        return yaw_bin(yaw_deg)

    # alternate names
    get_width_bin = width_px_bin
    get_char_height_bin = char_height_px_bin
    get_yaw_bin = yaw_bin

    def record_sample(
        self,
        plate_text: PlateText,
        plate_type: PlateType,
        illumination: str = "day_clear",
        plate_width_px: float = 100.0,
        char_height_px: float = 20.0,
        yaw_deg: float = 0.0,
        degradation_tag: str = "clean",
        readable_fields: dict[str, bool] | None = None,
    ) -> None:
        self.total_samples += 1
        # region (normalize alias)
        region = "袖ケ浦" if plate_text.region == "袖ヶ浦" else plate_text.region
        if region in self.region_counts:
            self.region_counts[region] += 1
        else:
            self.region_counts[plate_text.region] += 1

        self.plate_type_counts[plate_type.value if isinstance(plate_type, PlateType) else str(plate_type)] += 1

        # per-character: count all chars in ocr_string (region+class+kana+formatted serial)
        for ch in plate_text.ocr_string:
            if ch in self.char_counts:
                self.char_counts[ch] += 1
            else:
                # still track unseen chars
                self.char_counts[ch] += 1

        wb = width_px_bin(plate_width_px)
        yb = yaw_bin(yaw_deg)
        chb = char_height_px_bin(char_height_px)

        self._illuminations.add(illumination)
        self._degradations.add(degradation_tag)

        self._matrix_iwy[(illumination, wb, yb)] += 1

        if readable_fields is None:
            fields_to_count = FIELDS
        else:
            fields_to_count = [f for f, ok in readable_fields.items() if ok]
        for f in fields_to_count:
            self._matrix_fcd[(f, chb, degradation_tag)] += 1

    # properties for matrices in nested form
    def _nested_iwy(self) -> dict[str, dict[str, dict[str, int]]]:
        out: dict[str, dict[str, dict[str, int]]] = {}
        for ill in sorted(self._illuminations):
            out[ill] = {}
            for wb in WIDTH_PX_BINS:
                out[ill][wb] = {}
                for yb in YAW_BINS:
                    out[ill][wb][yb] = int(self._matrix_iwy.get((ill, wb, yb), 0))
        return out

    def _nested_fcd(self) -> dict[str, dict[str, dict[str, int]]]:
        out: dict[str, dict[str, dict[str, int]]] = {}
        for f in FIELDS:
            out[f] = {}
            for chb in CHAR_HEIGHT_PX_BINS:
                out[f][chb] = {}
                for deg in sorted(self._degradations):
                    out[f][chb][deg] = int(self._matrix_fcd.get((f, chb, deg), 0))
        return out

    @property
    def matrix_illumination_width_yaw(self) -> dict:
        return self._nested_iwy()

    @property
    def matrix_field_charheight_degradation(self) -> dict:
        return self._nested_fcd()

    # alias properties via methods
    def get_matrix_illumination_width_yaw(self) -> dict:
        return self._nested_iwy()

    def get_matrix_field_charheight_degradation(self) -> dict:
        return self._nested_fcd()

    def find_blind_spots(self, min_region_count: int = 1, min_cell_count: int = 1) -> dict[str, Any]:
        missing_regions = [r for r in self.region_names if self.region_counts.get(r, 0) < min_region_count]
        under_regions = missing_regions  # alias semantics
        missing_chars = [c for c in self.charset if self.char_counts.get(c, 0) < min_cell_count]
        # matrix cells
        blind_iwy: list[dict[str, Any]] = []
        blind_iwy_str: list[str] = []
        for ill in sorted(self._illuminations):
            for wb in WIDTH_PX_BINS:
                for yb in YAW_BINS:
                    cnt = int(self._matrix_iwy.get((ill, wb, yb), 0))
                    if cnt < min_cell_count:
                        blind_iwy.append({"illumination": ill, "width_px_bin": wb, "yaw_bin": yb, "count": cnt})
                        blind_iwy_str.append(f"{ill}|{wb}|{yb}")
        # If no samples yet, report all combos for a default illumination? Keep empty illuminations -> also flag?
        if not self._illuminations:
            for wb in WIDTH_PX_BINS:
                for yb in YAW_BINS:
                    blind_iwy.append({"illumination": "day_clear", "width_px_bin": wb, "yaw_bin": yb, "count": 0})
                    blind_iwy_str.append(f"day_clear|{wb}|{yb}")

        blind_fcd: list[dict[str, Any]] = []
        blind_fcd_str: list[str] = []
        degs = sorted(self._degradations) if self._degradations else ["clean"]
        for f in FIELDS:
            for chb in CHAR_HEIGHT_PX_BINS:
                for deg in degs:
                    cnt = int(self._matrix_fcd.get((f, chb, deg), 0))
                    if cnt < min_cell_count:
                        blind_fcd.append({"field": f, "char_height_px_bin": chb, "degradation_tag": deg, "count": cnt})
                        blind_fcd_str.append(f"{f}|{chb}|{deg}")

        result: dict[str, Any] = {
            "missing_regions": missing_regions,
            "underrepresented_regions": missing_regions,
            "low_count_regions": missing_regions,
            "regions": missing_regions,
            "missing_characters": missing_chars,
            "underrepresented_characters": missing_chars,
            "missing_chars": missing_chars,
            "characters": missing_chars,
            "charset_characters": missing_chars,
            "blind_cells_illumination_width_yaw": blind_iwy,
            "blind_spots_illumination_width_yaw": blind_iwy,
            "matrix_illumination_width_yaw_blind_cells": blind_iwy,
            "illumination_width_yaw": blind_iwy_str,
            "blind_cells_field_charheight_degradation": blind_fcd,
            "blind_spots_field_charheight_degradation": blind_fcd,
            "matrix_field_charheight_degradation_blind_cells": blind_fcd,
            "field_charheight_degradation": blind_fcd_str,
            "blind_cells": blind_iwy_str + blind_fcd_str,
        }
        return result

    def to_dict(self) -> dict[str, Any]:
        blind = self.find_blind_spots()
        return {
            "total_samples": self.total_samples,
            "sample_count": self.total_samples,
            "region_counts": {k: int(self.region_counts.get(k, 0)) for k in self.region_names},
            "char_counts": {k: int(self.char_counts.get(k, 0)) for k in self.charset},
            "character_counts": {k: int(self.char_counts.get(k, 0)) for k in self.charset},
            "plate_type_counts": dict(self.plate_type_counts),
            "width_px_bins": list(WIDTH_PX_BINS),
            "char_height_px_bins": list(CHAR_HEIGHT_PX_BINS),
            "yaw_bins": list(YAW_BINS),
            "illuminations": sorted(self._illuminations),
            "degradation_tags": sorted(self._degradations),
            "illumination_width_yaw": self._nested_iwy(),
            "matrix_illumination_width_yaw": self._nested_iwy(),
            "illumination_x_width_x_yaw": self._nested_iwy(),
            "field_charheight_degradation": self._nested_fcd(),
            "matrix_field_charheight_degradation": self._nested_fcd(),
            "field_x_charheight_x_degradation": self._nested_fcd(),
            "blind_spots": blind,
            "covered_regions": sum(1 for r in self.region_names if self.region_counts.get(r, 0) > 0),
            "covered_characters": sum(1 for c in self.charset if self.char_counts.get(c, 0) > 0),
        }

    def to_markdown(self) -> str:
        d = self.to_dict()
        blind = d["blind_spots"]
        covered_r = d["covered_regions"]
        covered_c = d["covered_characters"]
        lines: list[str] = []
        lines.append("# カバレッジレポート")
        lines.append("")
        lines.append(f"総サンプル数: {self.total_samples}")
        lines.append("")
        lines.append("## 概要")
        lines.append("")
        lines.append("| 項目 | カバー済み | 全体 |")
        lines.append("| --- | ---: | ---: |")
        lines.append(f"| 地域 | {covered_r} | {len(self.region_names)} |")
        lines.append(f"| 文字 | {covered_c} | {len(self.charset)} |")
        lines.append(f"| サンプル数 | {self.total_samples} | - |")
        lines.append("")
        lines.append("## 地域別カウント")
        lines.append("")
        lines.append("| 地域 | 件数 |")
        lines.append("| --- | ---: |")
        for r in self.region_names:
            lines.append(f"| {r} | {int(self.region_counts.get(r,0))} |")
        lines.append("")
        lines.append("## 照明 × 幅 × ヨー (3Dクロス集計)")
        lines.append("")
        lines.append("| 照明 | 幅ビン | ヨービン | 件数 |")
        lines.append("| --- | --- | --- | ---: |")
        nested = self._nested_iwy()
        if nested:
            for ill, wb_map in nested.items():
                for wb, yb_map in wb_map.items():
                    for yb, cnt in yb_map.items():
                        lines.append(f"| {ill} | {wb} | {yb} | {cnt} |")
        else:
            lines.append("| day_clear | <40px | front_0_20 | 0 |")
        lines.append("")
        lines.append("## フィールド × 文字高 × 劣化 (3Dクロス集計)")
        lines.append("")
        lines.append("| フィールド | 文字高ビン | 劣化タグ | 件数 |")
        lines.append("| --- | --- | --- | ---: |")
        nested2 = self._nested_fcd()
        for f, ch_map in nested2.items():
            for chb, deg_map in ch_map.items():
                for deg, cnt in deg_map.items():
                    lines.append(f"| {f} | {chb} | {deg} | {cnt} |")
        lines.append("")
        lines.append("## ブラインドスポット")
        lines.append("")
        lines.append(f"未カバー地域 ({len(blind['missing_regions'])}件): " + (", ".join(blind["missing_regions"][:50]) if blind["missing_regions"] else "なし"))
        lines.append("")
        lines.append(f"未カバー文字 ({len(blind['missing_characters'])}件): " + (", ".join(blind["missing_characters"][:80]) if blind["missing_characters"] else "なし"))
        lines.append("")
        lines.append(f"未カバーセル（照明×幅×ヨー）: {len(blind['blind_cells_illumination_width_yaw'])}件")
        lines.append("")
        lines.append(f"未カバーセル（フィールド×文字高×劣化）: {len(blind['blind_cells_field_charheight_degradation'])}件")
        lines.append("")
        if blind["missing_regions"]:
            lines.append("### 未カバー地域一覧")
            lines.append("")
            for r in blind["missing_regions"]:
                lines.append(f"- {r}")
            lines.append("")
        return "\n".join(lines)

    def save_report(self, json_path: Path, md_path: Path | None = None) -> None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        if md_path is not None:
            md_path = Path(md_path)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(self.to_markdown())
