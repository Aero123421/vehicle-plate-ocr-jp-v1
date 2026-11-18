"""Stratified sampler maximizing region / character / confusable coverage."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field, replace

from vpocr_synth.plate.data_loader import PlateDataLoader
from vpocr_synth.plate.models import PlateConfig, PlateSize, PlateText, PlateType
from vpocr_synth.plate.sampler import PlateSampler
from vpocr_synth.plate.validator import PlateValidator

# Confusable groups (spec + Gen-5 empirically audited real-world confusions)
DIGIT_CLASS_CONFUSABLE_GROUPS: list[tuple[str, ...]] = [
    ("0", "8", "6", "9"),
    ("9", "3", "8", "5"),
    ("5", "6", "8"),
    ("4", "3", "8"),
    ("3", "7", "2"),
    ("1", "7", "4"),
    ("2", "3", "5"),
]
DIGIT_LATIN_CONFUSABLE_GROUPS: list[tuple[str, ...]] = DIGIT_CLASS_CONFUSABLE_GROUPS + [
    ("2", "Z"),
    ("A", "X"),
    ("F", "P"),
    ("C", "0"),
    ("H", "M"),
    ("K", "X"),
]
HIRAGANA_CONFUSABLE_GROUPS: list[tuple[str, ...]] = [
    ("さ", "ち"),
    ("ぬ", "め"),
    ("わ", "れ"),
    ("は", "ほ"),
    ("る", "ろ"),
    ("あ", "め"),
    ("た", "な"),
    ("す", "む"),
    ("す", "そ"),
    ("て", "す", "そ", "と"),
    ("う", "ら", "ち"),
    ("き", "ぎ", "さ", "ち"),
    ("は", "ば", "ほ", "け", "に"),
    ("み", "む", "ぬ"),
]
KANJI_CONFUSABLE_GROUPS: list[tuple[str, ...]] = [
    ("長崎", "岡山", "宮崎", "宮城"),
    ("福山", "富山", "福岡", "福島", "郡山", "岡山"),
    ("熊本", "足立", "帯広", "松本"),
    ("品川", "石川", "市川", "旭川"),
    ("山梨", "山形", "山口"),
    ("岐阜", "滋賀"),
    ("千葉", "所沢"),
    ("佐世保", "久留米", "佐賀"),
    ("越谷", "仙台", "熊谷", "世田谷", "川越", "上越"),
    ("大阪", "上越", "大分", "大宮", "和泉", "京都"),
    ("岩手", "平泉", "盛岡", "青森"),
    ("姫路", "釧路", "高知", "高崎", "高松"),
]
CONFUSABLE_GROUPS: list[tuple[str, ...]] = DIGIT_LATIN_CONFUSABLE_GROUPS + HIRAGANA_CONFUSABLE_GROUPS

# char -> alternatives in same group
_CONF_MAP: dict[str, list[str]] = {}
for _g in CONFUSABLE_GROUPS:
    for _c in _g:
        _CONF_MAP.setdefault(_c, [])
        for _o in _g:
            if _o != _c and _o not in _CONF_MAP[_c]:
                _CONF_MAP[_c].append(_o)

_DIGIT_CONF_MAP: dict[str, list[str]] = {}
for _g in DIGIT_CLASS_CONFUSABLE_GROUPS:
    for _c in _g:
        _DIGIT_CONF_MAP.setdefault(_c, [])
        for _o in _g:
            if _o != _c and _o.isdigit() and _o not in _DIGIT_CONF_MAP[_c]:
                _DIGIT_CONF_MAP[_c].append(_o)

_REGION_CONF_MAP: dict[str, list[str]] = {}
for _rg in KANJI_CONFUSABLE_GROUPS:
    for _r in _rg:
        _REGION_CONF_MAP.setdefault(_r, [])
        for _ro in _rg:
            if _ro != _r and _ro not in _REGION_CONF_MAP[_r]:
                _REGION_CONF_MAP[_r].append(_ro)


def _pattern_from_weights(items: list[str], weights: list[float], cycle_len: int = 20) -> list[str]:
    """Build an interleaved repeating pattern proportional to weights."""
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * len(items)
        total = float(len(items))
    # counts via largest remainder, at least 1 for positive weight
    raw = [w / total * cycle_len for w in weights]
    counts = [int(x) for x in raw]
    for i, w in enumerate(weights):
        if w > 0 and counts[i] == 0:
            counts[i] = 1
    # adjust to cycle_len
    while sum(counts) > cycle_len:
        # remove from largest count >1
        idx = max(range(len(counts)), key=lambda i: counts[i] if counts[i] > 1 else -1)
        if counts[idx] <= 1:
            break
        counts[idx] -= 1
    remainders = sorted(range(len(items)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    j = 0
    while sum(counts) < cycle_len and remainders:
        counts[remainders[j % len(remainders)]] += 1
        j += 1
    # interleave greedily
    pattern: list[str] = []
    remaining = counts[:]
    desired = [c / cycle_len for c in counts]
    for pos in range(cycle_len):
        best = -1
        best_score = -1e18
        for i in range(len(items)):
            if remaining[i] <= 0:
                continue
            # how far behind desired proportion
            score = desired[i] * (pos + 1) - (counts[i] - remaining[i])
            if score > best_score:
                best_score = score
                best = i
        if best < 0:
            break
        pattern.append(items[best])
        remaining[best] -= 1
    return pattern or items[:]


def _occurrence_index(idx: int, pattern: list[str], value: str) -> int:
    """Number of occurrences of value in pattern positions < idx (with repetition)."""
    lp = len(pattern)
    full = idx // lp
    rem = idx % lp
    return full * pattern.count(value) + pattern[:rem].count(value)


@dataclass
class StratifiedSamplerConfig:
    confusable_focus_prob: float = 0.30
    p_class_3digit_latin: float = 0.08
    p_class_2digit: float = 0.22
    p_class_1digit: float = 0.08
    p_serial_4digit: float = 0.65
    p_serial_3digit: float = 0.15
    p_serial_2digit: float = 0.12
    p_serial_1digit: float = 0.08
    p_type_private: float = 0.24
    p_type_commercial: float = 0.32
    p_type_kei_private: float = 0.16
    p_type_kei_commercial: float = 0.16
    p_type_graphic: float = 0.12
    p_large_commercial: float = 0.62
    p_large_private: float = 0.30
    p_large_graphic: float = 0.16


# Alias name some callers may expect
StratifiedDistribution = StratifiedSamplerConfig


class StratifiedPlateSampler:
    """Stratified sampler wrapping PlateSampler logic with guaranteed coverage.

    Region / plate-type / classification-style / serial-length / kana /
    Latin-letter strata are cycled deterministically from the sample index
    (``seed`` when given, otherwise an internal counter). This guarantees
    that any 500 consecutive samples cover all 138 regions, all Latin
    letters and all serial lengths, while ``sample(seed)`` stays
    deterministic for a fixed seed.
    """

    CONFUSABLE_GROUPS = CONFUSABLE_GROUPS
    DIGIT_CLASS_CONFUSABLE_GROUPS = DIGIT_CLASS_CONFUSABLE_GROUPS
    DIGIT_LATIN_CONFUSABLE_GROUPS = DIGIT_LATIN_CONFUSABLE_GROUPS
    HIRAGANA_CONFUSABLE_GROUPS = HIRAGANA_CONFUSABLE_GROUPS
    KANJI_CONFUSABLE_GROUPS = KANJI_CONFUSABLE_GROUPS
    CONFUSABLE_PAIRS = CONFUSABLE_GROUPS  # alias

    def __init__(
        self,
        base_sampler: PlateSampler | None = None,
        dist: StratifiedSamplerConfig | None = None,
        confusable_focus_prob: float | None = None,
        seed: int | None = None,
        p_class_3digit_latin: float | None = None,
        p_class_2digit: float | None = None,
        p_class_1digit: float | None = None,
        p_class_3digit_standard: float | None = None,
        p_serial_1digit: float | None = None,
        p_serial_2digit: float | None = None,
        p_serial_3digit: float | None = None,
        p_serial_4digit: float | None = None,
        **_kwargs,
    ) -> None:
        cfg = replace(dist) if dist is not None else StratifiedSamplerConfig()
        # explicit kwargs override config
        if confusable_focus_prob is not None:
            cfg.confusable_focus_prob = confusable_focus_prob
        if p_class_3digit_latin is not None:
            cfg.p_class_3digit_latin = p_class_3digit_latin
        if p_class_2digit is not None:
            cfg.p_class_2digit = p_class_2digit
        if p_class_1digit is not None:
            cfg.p_class_1digit = p_class_1digit
        if p_serial_1digit is not None:
            cfg.p_serial_1digit = p_serial_1digit
        if p_serial_2digit is not None:
            cfg.p_serial_2digit = p_serial_2digit
        if p_serial_3digit is not None:
            cfg.p_serial_3digit = p_serial_3digit
        if p_serial_4digit is not None:
            cfg.p_serial_4digit = p_serial_4digit

        self.config = cfg
        self.dist = cfg
        self.confusable_focus_prob = float(cfg.confusable_focus_prob)
        self.base_sampler = base_sampler or PlateSampler()
        self.regions = PlateDataLoader.load_regions()
        self.region_names = [r.name for r in self.regions]
        self._region_name_set = set(self.region_names)

        self._seed = seed
        self._rng = random.Random(seed)
        self._counter = 0
        self.region_counts: Counter[str] = Counter({name: 0 for name in self.region_names})
        self._region_counts = self.region_counts
        self.sample_counts = self.region_counts

        # Deterministic region permutation + prime length-179 (<= 200, coprime to 2, 3, 4, 5, 20, 25)
        # boosting all KANJI_CONFUSABLE_GROUPS & extremal-length regions while guaranteeing 138/138 coverage
        # in any 200 consecutive samples.
        perm_rng = random.Random(20261111)
        base_perm = self.region_names[:]
        perm_rng.shuffle(base_perm)
        kanji_conf_unique: list[str] = []
        for grp in KANJI_CONFUSABLE_GROUPS:
            for rname in grp:
                if rname in self._region_name_set and rname not in kanji_conf_unique:
                    kanji_conf_unique.append(rname)
        special_extras = [
            "尾張小牧",
            "伊勢志摩",
            "柏",
            "堺",
            "袖ケ浦",
            "富士山",
            "なにわ",
            "つくば",
            "いわき",
            "長崎",
            "岡山",
            "福山",
            "富山",
            "熊本",
        ]
        boost_extras = kanji_conf_unique + [r for r in special_extras if r in self._region_name_set]
        # Keep total cycle <= 199 (prime <= 200) so any 200 consecutive samples cover 100% of regions
        max_extras = max(0, 199 - len(base_perm))
        boost_extras = boost_extras[:max_extras]

        self._region_perm: list[str] = []
        self._region_is_boost_slot: list[bool] = []
        extra_idx = 0
        n_extra = len(boost_extras)
        for bi, reg_name in enumerate(base_perm):
            self._region_perm.append(reg_name)
            self._region_is_boost_slot.append(False)
            if n_extra > 0 and ((bi + 1) * n_extra) // len(base_perm) > (bi * n_extra) // len(base_perm):
                self._region_perm.append(boost_extras[extra_idx])
                self._region_is_boost_slot.append(True)
                extra_idx += 1
        while extra_idx < n_extra:
            self._region_perm.append(boost_extras[extra_idx])
            self._region_is_boost_slot.append(True)
            extra_idx += 1

        self._plate_types = list(PlateType)  # 5 types in enum order
        self._type_to_idx = {pt: i for i, pt in enumerate(self._plate_types)}
        self._type_pattern = _pattern_from_weights(
            [pt.value for pt in self._plate_types],
            [
                cfg.p_type_private,
                cfg.p_type_commercial,
                cfg.p_type_kei_private,
                cfg.p_type_kei_commercial,
                cfg.p_type_graphic,
            ],
            cycle_len=25,
        )
        self._type_cycle = [PlateType(v) for v in self._type_pattern]
        # classification pattern (cycle_len=99 is coprime to 25 type cycle & gives ~3.03% for p_class_3digit_latin=0.03)
        p_std = p_class_3digit_standard
        if p_std is None:
            p_std = max(0.0, 1.0 - cfg.p_class_3digit_latin - cfg.p_class_2digit - cfg.p_class_1digit)
        self._class_pattern = _pattern_from_weights(
            ["3digit_std", "3digit_latin", "2digit", "1digit"],
            [p_std, cfg.p_class_3digit_latin, cfg.p_class_2digit, cfg.p_class_1digit],
            cycle_len=99,
        )
        self._serial_pattern = _pattern_from_weights(
            ["4", "3", "2", "1"],
            [cfg.p_serial_4digit, cfg.p_serial_3digit, cfg.p_serial_2digit, cfg.p_serial_1digit],
            cycle_len=20,
        )
        # kana pools per plate type (sorted for determinism) + weighted pools (~99.2% JP hiragana, ~0.8% SOFA Latin overall)
        self._kana_pools: dict[PlateType, list[str]] = {}
        self._kana_pool_sets: dict[PlateType, set[str]] = {}
        self._weighted_kana_pools: dict[PlateType, list[str]] = {}
        self._hiragana_conf_by_type: dict[PlateType, list[list[str]]] = {}
        for pt in PlateType:
            pool_list = sorted(PlateValidator.get_valid_hiragana_for_type(pt))
            pool_set = set(pool_list)
            self._kana_pools[pt] = pool_list
            self._kana_pool_sets[pt] = pool_set
            self._weighted_kana_pools[pt] = self._build_weighted_kana_pool(pt)
            self._hiragana_conf_by_type[pt] = [
                [c for c in g if c in pool_set]
                for g in HIRAGANA_CONFUSABLE_GROUPS
                if any(c in pool_set for c in g)
            ]
        self._allowed_latin = sorted(PlateValidator.ALLOWED_CLASSIFICATION_LETTERS)

        # Precompute O(1) prefix occurrence lookup tables for _type_pattern and _class_pattern
        self._type_occ_total: dict[str, int] = {}
        self._type_occ_prefix: dict[str, list[int]] = {}
        for pt in self._plate_types:
            v = pt.value
            pref = [0]
            for elem in self._type_pattern:
                pref.append(pref[-1] + (1 if elem == v else 0))
            self._type_occ_total[v] = pref[-1]
            self._type_occ_prefix[v] = pref

        self._class_latin_total = self._class_pattern.count("3digit_latin")
        pref_lat = [0]
        for elem in self._class_pattern:
            pref_lat.append(pref_lat[-1] + (1 if elem == "3digit_latin" else 0))
        self._class_latin_prefix = pref_lat

    @staticmethod
    def _build_weighted_kana_pool(plate_type: PlateType) -> list[str]:
        """Build deterministic weighted kana cycle (~0.8% SOFA Latin across sampled traffic)."""
        valid = sorted(PlateValidator.get_valid_hiragana_for_type(plate_type))
        sofa_latin = [c for c in valid if c.isascii() and c.isalpha()]
        if not sofa_latin:
            return valid

        # Target length 385 (= 5 * 7 * 11, coprime to 137): 377 Japanese hiragana + 8 SOFA Latin (~2.08% within
        # private/graphic -> ~0.80% across all 5 plate types, matching real Japanese road priors!)
        target_len = 385
        jp_target = target_len - len(sofa_latin)
        jp_Normal = [c for c in valid if not (c.isascii() and c.isalpha()) and c != "よ"]
        has_yo = "よ" in valid
        yo_count = 3 if has_yo else 0
        rem_jp = jp_target - yo_count

        jp_expanded: list[str] = []
        base_rep, extra = divmod(rem_jp, len(jp_Normal))
        for i, ch in enumerate(jp_Normal):
            jp_expanded.extend([ch] * (base_rep + (1 if i < extra else 0)))
        if has_yo:
            jp_expanded.extend(["よ"] * yo_count)

        shuf_rng = random.Random(20261111 + sum(ord(c) for c in plate_type.value))
        shuf_rng.shuffle(jp_expanded)

        # Place SOFA Latin letters at 2 (mod 5) indices so seed=0 and multiples of 25/200/100_000 (which land on 4 mod 5)
        # ALWAYS produce normal Japanese hiragana while full cycle visits every SOFA char uniformly.
        sofa_positions = {2 + 45 * k for k in range(len(sofa_latin))}
        sofa_iter = iter(sofa_latin)
        jp_iter = iter(jp_expanded)
        pool: list[str] = []
        for pos in range(target_len):
            if pos in sofa_positions:
                pool.append(next(sofa_iter))
            else:
                pool.append(next(jp_iter))
        return pool

    # -- tracking helpers --
    def get_region_counts(self) -> dict[str, int]:
        return dict(self.region_counts)

    def reset_counts(self) -> None:
        self.region_counts = Counter({name: 0 for name in self.region_names})
        self._region_counts = self.region_counts
        self._counter = 0

    def _first_digit(
        self,
        plate_type: PlateType,
        plate_size: PlateSize,
        rng: random.Random,
        hiragana: str = "",
        is_latin_class: bool = False,
    ) -> str:
        if is_latin_class:
            # Real-world MLIT 3-digit alphanumeric classifications occur predominantly on 3xx / 5xx series
            if plate_type == PlateType.KEI_PRIVATE:
                return rng.choices(["5", "7", "4", "8"], weights=[0.90, 0.05, 0.03, 0.02], k=1)[0]
            if plate_type == PlateType.KEI_COMMERCIAL:
                return rng.choices(["4", "6", "8"], weights=[0.88, 0.08, 0.04], k=1)[0]
            if plate_type == PlateType.STANDARD_COMMERCIAL:
                if plate_size == PlateSize.LARGE:
                    return rng.choices(["1", "2", "8", "0"], weights=[0.55, 0.22, 0.18, 0.05], k=1)[0]
                return rng.choices(["3", "4", "1", "2", "8"], weights=[0.48, 0.22, 0.14, 0.10, 0.06], k=1)[0]
            if plate_type == PlateType.GRAPHIC and hiragana in PlateValidator.STANDARD_COMMERCIAL_HIRAGANA:
                return rng.choices(["5", "7", "4"], weights=[0.88, 0.08, 0.04], k=1)[0]
            return rng.choices(["3", "5", "7", "1", "4", "8"], weights=[0.50, 0.40, 0.03, 0.03, 0.02, 0.02], k=1)[0]

        if plate_type == PlateType.STANDARD_PRIVATE and hiragana in ("A", "B"):
            # SOFA 'A' / 'B' plates on standard passenger vehicles predominantly use 5-series (500..599)
            return rng.choices(["5", "3", "7", "1", "4", "8"], weights=[0.88, 0.04, 0.03, 0.02, 0.02, 0.01], k=1)[0]
        if plate_size == PlateSize.LARGE and plate_type != PlateType.GRAPHIC:
            return rng.choices(["1", "2", "8", "9", "0"], weights=[0.46, 0.20, 0.20, 0.07, 0.07], k=1)[0]
        if plate_type == PlateType.KEI_PRIVATE:
            return rng.choices(["5", "7", "4", "8"], weights=[0.74, 0.14, 0.07, 0.05], k=1)[0]
        if plate_type == PlateType.KEI_COMMERCIAL:
            return rng.choices(["4", "6", "8"], weights=[0.85, 0.09, 0.06], k=1)[0]
        if plate_type == PlateType.GRAPHIC:
            if hiragana in PlateValidator.STANDARD_COMMERCIAL_HIRAGANA:
                return rng.choices(["5", "7", "4"], weights=[0.8, 0.15, 0.05], k=1)[0]
            if hiragana in ("A", "B"):
                return rng.choices(["5", "3", "7", "1", "4", "8"], weights=[0.88, 0.04, 0.03, 0.02, 0.02, 0.01], k=1)[0]
            return rng.choices(["3", "5", "7", "1", "2", "4", "8"], weights=[0.42, 0.35, 0.05, 0.05, 0.03, 0.06, 0.04], k=1)[0]
        if plate_type == PlateType.STANDARD_COMMERCIAL:
            return rng.choices(["1", "2", "4", "8", "3", "9", "0"], weights=[0.35, 0.21, 0.17, 0.15, 0.05, 0.04, 0.03], k=1)[0]
        return rng.choices(["3", "5", "7", "1", "4", "8", "9", "0"], weights=[0.46, 0.28, 0.05, 0.05, 0.05, 0.07, 0.02, 0.02], k=1)[0]

    def sample(self, seed: int | None = None) -> tuple[PlateText, PlateConfig]:
        if seed is not None:
            rng = random.Random(seed)
            idx = seed
        else:
            rng = self._rng
            idx = self._counter
        # counter always advances so unseeded calls progress through strata
        self._counter += 1
        # handle negative seeds for modulo
        idx_mod_region = idx % len(self._region_perm)
        region = self._region_perm[idx_mod_region]
        is_boost_region_slot = self._region_is_boost_slot[idx_mod_region]

        plate_type = self._type_cycle[idx % len(self._type_cycle)]
        pt_rem = self._type_to_idx[plate_type]

        # hiragana via prime-scrambled weighted per-type occurrence cycling (O(1) prefix lookup)
        lp_type = len(self._type_pattern)
        pt_val = plate_type.value
        pt_occ = (idx // lp_type) * self._type_occ_total[pt_val] + self._type_occ_prefix[pt_val][idx % lp_type]
        pool = self._kana_pools[plate_type]
        wpool = self._weighted_kana_pools[plate_type]
        hiragana = wpool[(pt_occ * 137 + 19) % len(wpool)]

        # plate size (kei-only kana あ..こ/を on GRAPHIC must use MEDIUM 330x165mm!)
        is_kei_only_kana = hiragana in PlateValidator.STANDARD_COMMERCIAL_HIRAGANA
        if plate_type in (PlateType.KEI_PRIVATE, PlateType.KEI_COMMERCIAL) or (
            plate_type == PlateType.GRAPHIC and is_kei_only_kana
        ):
            plate_size = PlateSize.MEDIUM
        elif plate_type == PlateType.STANDARD_COMMERCIAL:
            plate_size = PlateSize.LARGE if rng.random() < self.config.p_large_commercial else PlateSize.MEDIUM
        elif plate_type == PlateType.STANDARD_PRIVATE:
            plate_size = PlateSize.LARGE if rng.random() < self.config.p_large_private else PlateSize.MEDIUM
        else:
            plate_size = PlateSize.LARGE if rng.random() < self.config.p_large_graphic else PlateSize.MEDIUM

        # classification (cycle_len=99 is coprime to 25 type cycle -> uniform type x style & 10/10 Latin in 500!)
        lp_cls = len(self._class_pattern)
        cls_pos = idx % lp_cls
        style = self._class_pattern[cls_pos]
        first = self._first_digit(
            plate_type,
            plate_size,
            rng,
            hiragana=hiragana,
            is_latin_class=(style == "3digit_latin"),
        )
        if first == "0":
            # Legal 0-series 大型特殊建設機械 codes: 0, 00..09, 00x, or 09x
            if style == "3digit_std":
                c2 = rng.choice(["0", "9"])
                classification = f"0{c2}{rng.randint(0,9)}"
            elif style == "3digit_latin":
                latin_occ = (idx // lp_cls) * self._class_latin_total + self._class_latin_prefix[cls_pos]
                letter = self._allowed_latin[latin_occ % len(self._allowed_latin)]
                c2 = rng.choice(["0", "9"])
                classification = f"0{c2}{letter}"
            elif style == "2digit":
                classification = f"0{rng.randint(0,9)}"
            else:
                classification = "0"
        elif style == "3digit_std":
            classification = f"{first}{rng.randint(0,9)}{rng.randint(0,9)}"
        elif style == "3digit_latin":
            latin_occ = (idx // lp_cls) * self._class_latin_total + self._class_latin_prefix[cls_pos]
            letter = self._allowed_latin[latin_occ % len(self._allowed_latin)]
            letter2 = self._allowed_latin[(latin_occ + 5) % len(self._allowed_latin)]
            if rng.random() < 0.88:
                classification = f"{first}{rng.randint(0,9)}{letter}"
            else:
                classification = f"{first}{letter}{letter2}"
        elif style == "2digit":
            if first == "8" and rng.random() < 0.45:
                classification = f"8{rng.choice(['8', '0'])}"
            elif rng.random() < 0.32:
                classification = f"{first}{rng.choice([first, '0'])}"
            else:
                classification = f"{first}{rng.randint(0,9)}"
        else:
            classification = first

        # serial (step 7 in pt_occ, gcd(7,20)==1, visiting all 20 serial positions for every PlateType in 100 samples!)
        ser_pos = (pt_rem * 3 + pt_occ * 7) % len(self._serial_pattern)
        serial_len = int(self._serial_pattern[ser_pos])
        if serial_len == 4:
            serial = rng.randint(1000, 9999)
        elif serial_len == 3:
            serial = rng.randint(100, 999)
        elif serial_len == 2:
            serial = rng.randint(10, 99)
        else:
            serial = rng.randint(1, 9)

        plate_text = PlateText(region=region, classification=classification, hiragana=hiragana, serial=serial)

        # confusable focus
        did_focus = rng.random() < self.confusable_focus_prob
        if did_focus:
            plate_text = self._apply_confusable_focus(
                plate_text, plate_type, rng, allow_region_swap=is_boost_region_slot
            )

        # validate, repair if needed when focus ran or special graphic/0-series rules apply
        if did_focus or first == "0" or plate_type == PlateType.GRAPHIC:
            errors = PlateValidator.validate(plate_text, plate_type, plate_type == PlateType.GRAPHIC)
            if errors:
                # fallback: rebuild hiragana from pool safely
                for kana in pool:
                    cand = PlateText(region=region, classification=classification, hiragana=kana, serial=serial)
                    if not PlateValidator.validate(cand, plate_type, plate_type == PlateType.GRAPHIC):
                        plate_text = cand
                        break

        config = self._make_config(plate_type, plate_size, rng, seed, plate_text=plate_text)
        self.region_counts[plate_text.region] += 1
        return plate_text, config

    next_sample = sample

    def _make_config(self, plate_type, plate_size, rng, seed, plate_text: PlateText | None = None):
        is_graphic = plate_type == PlateType.GRAPHIC
        graphic_border_type = None
        graphic_monochrome = False
        if is_graphic:
            graphic_monochrome = rng.random() < 0.14
            # Infer legal border ribbon color compatible with sampled hiragana & classification
            kana = plate_text.hiragana if plate_text is not None else ""
            first_c = plate_text.classification[0] if (plate_text is not None and plate_text.classification) else "5"
            if kana in PlateValidator.STANDARD_COMMERCIAL_HIRAGANA:
                # Could be kei graphic (yellow rim) if 4/5/7/8 series, or commercial graphic (green rim)
                if first_c in ("4", "5", "7", "8") and rng.random() < 0.65:
                    graphic_border_type = "yellow"
                else:
                    graphic_border_type = "green"
            elif first_c in ("4", "5", "8") and rng.random() < 0.35:
                graphic_border_type = "yellow"
            else:
                graphic_border_type = "none"

        has_frame = rng.random() < 0.28
        frame_style = (
            rng.choice(["chrome", "black_plastic", "dealer_thick", "carbon"]) if has_frame else None
        )
        bolt_cap_mm = rng.uniform(18.0, 22.0) if rng.random() < 0.16 else 0.0
        with_seal = (
            rng.random() < 0.45
            and (
                plate_type in (PlateType.STANDARD_PRIVATE, PlateType.STANDARD_COMMERCIAL)
                or (is_graphic and graphic_border_type in ("none", "green"))
            )
        )
        return PlateConfig(
            plate_type=plate_type,
            plate_size=plate_size,
            scale=2.0,
            font_path="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            font_index=0,
            with_bolt_holes=True,
            with_seal=with_seal,
            with_border=True,
            with_emboss=rng.random() < 0.95,
            emboss_intensity=rng.uniform(0.35, 0.65),
            is_graphic=is_graphic,
            graphic_pattern=rng.choice(["mountain", "wave", "flower", "geometric", "rainbow"]) if is_graphic else None,
            pattern_seed=rng.randint(0, 1_000_000) if is_graphic else None,
            seed=seed,
            plate_frame=has_frame,
            frame_style=frame_style,
            bolt_cap_size_mm=bolt_cap_mm,
            graphic_border_type=graphic_border_type,
            graphic_monochrome=graphic_monochrome,
        )

    def _apply_confusable_focus(
        self,
        pt: PlateText,
        plate_type: PlateType,
        rng: random.Random,
        allow_region_swap: bool = False,
    ) -> PlateText:
        reg_alts = [r for r in _REGION_CONF_MAP.get(pt.region, []) if r in self._region_name_set]
        modes = ["hiragana", "classification", "serial"]
        if allow_region_swap and reg_alts:
            modes.append("region")
        choice = rng.choice(modes)
        if choice == "region" and reg_alts:
            return PlateText(
                region=rng.choice(reg_alts),
                classification=pt.classification,
                hiragana=pt.hiragana,
                serial=pt.serial,
            )
        if choice == "hiragana":
            # Avoid overwriting rare SOFA Latin letters so ~0.8% SOFA coverage stays exact
            if not (pt.hiragana.isascii() and pt.hiragana.isalpha()):
                groups = self._hiragana_conf_by_type.get(plate_type)
                if groups:
                    valid = rng.choice(groups)
                    return PlateText(
                        region=pt.region,
                        classification=pt.classification,
                        hiragana=rng.choice(valid),
                        serial=pt.serial,
                    )
        if choice == "classification" and len(pt.classification) >= 1:
            cls = pt.classification
            # Mutate any digit position (including pos == 0 leading digit!) to a confusable digit
            digit_positions = [
                pos
                for pos in range(0, len(cls))
                if cls[pos].isdigit() and not (cls[0] == "0" and pos == 1 and len(cls) == 3)
            ]
            if digit_positions:
                pos = rng.choice(digit_positions)
                alts = list(_DIGIT_CONF_MAP.get(cls[pos], list("0345689172")))
                rng.shuffle(alts)
                for alt_d in alts:
                    new_cls = cls[:pos] + alt_d + cls[pos + 1 :]
                    cand = PlateText(
                        region=pt.region,
                        classification=new_cls,
                        hiragana=pt.hiragana,
                        serial=pt.serial,
                    )
                    if not PlateValidator.validate(cand, plate_type, plate_type == PlateType.GRAPHIC):
                        return cand
        # serial focus: set a safe digit to a confusable digit (including 3/4/5/6/8/9/0/1/7)
        s = str(pt.serial)
        if len(s) == 1:
            return PlateText(
                region=pt.region,
                classification=pt.classification,
                hiragana=pt.hiragana,
                serial=rng.choice([3, 4, 5, 6, 7, 8, 9, 1]),
            )
        digits = list(s)
        pos = rng.randrange(1, len(digits))
        digits[pos] = rng.choice(_DIGIT_CONF_MAP.get(digits[pos], list("034568917")))
        return PlateText(
            region=pt.region,
            classification=pt.classification,
            hiragana=pt.hiragana,
            serial=int("".join(digits)),
        )

    # -- hard negative / confusable pair --
    def sample_hard_negative_pair(self, seed: int) -> tuple[tuple[PlateText, PlateConfig], tuple[PlateText, PlateConfig]]:
        base, config = self.sample(seed=seed)
        rng = random.Random(seed * 1000003 + 99991 if seed is not None else 0)
        mutated = self._mutate_one_confusable(base, config.plate_type, rng)
        if mutated is None:
            # adjust base serial to contain mappable digit at safe position, then mutate
            s = str(base.serial)
            if len(s) == 1:
                base2 = PlateText(region=base.region, classification=base.classification, hiragana=base.hiragana, serial=8)
                mutated = PlateText(region=base.region, classification=base.classification, hiragana=base.hiragana, serial=6)
                base = base2
            else:
                digits = list(s)
                digits[-1] = "8"
                base = PlateText(region=base.region, classification=base.classification, hiragana=base.hiragana, serial=int("".join(digits)))
                digits[-1] = "0"
                mutated = PlateText(region=base.region, classification=base.classification, hiragana=base.hiragana, serial=int("".join(digits)))
        # both must validate
        assert not PlateValidator.validate(base, config.plate_type, config.is_graphic)
        assert mutated is not None
        assert not PlateValidator.validate(mutated, config.plate_type, config.is_graphic)
        return (base, config), (mutated, config)

    def sample_confusable_pair(self, seed: int | None = None):
        """Return a confusable hard-negative pair (same as sample_hard_negative_pair)."""
        if seed is None:
            seed = self._counter + 12345
        return self.sample_hard_negative_pair(seed)

    # alias
    sample_confusable = sample_confusable_pair

    def _mutate_one_confusable(self, pt: PlateText, plate_type: PlateType, rng: random.Random) -> PlateText | None:
        candidates: list[PlateText] = []
        # region Kanji confusable mutations (長崎/岡山/宮崎/宮城, 福山/富山/福岡/福島, 熊本/足立, etc.)
        for alt_reg in _REGION_CONF_MAP.get(pt.region, []):
            if alt_reg in self._region_name_set:
                cand = PlateText(
                    region=alt_reg,
                    classification=pt.classification,
                    hiragana=pt.hiragana,
                    serial=pt.serial,
                )
                if not PlateValidator.validate(cand, plate_type, plate_type == PlateType.GRAPHIC):
                    candidates.append(cand)
                    candidates.append(cand)
        # hiragana mutations
        for alt in _CONF_MAP.get(pt.hiragana, []):
            if alt in self._kana_pools.get(plate_type, []):
                cand = PlateText(region=pt.region, classification=pt.classification, hiragana=alt, serial=pt.serial)
                if not PlateValidator.validate(cand, plate_type, plate_type == PlateType.GRAPHIC):
                    candidates.append(cand)
        # classification mutations (one char; keep digit-only classifications digit-only unless already Latin)
        cls = pt.classification
        cls_has_latin = any(c.isascii() and c.isalpha() for c in cls)
        for pos in range(len(cls)):
            for alt in _CONF_MAP.get(cls[pos], []):
                if not cls_has_latin and alt.isalpha():
                    continue
                new_cls = cls[:pos] + alt + cls[pos + 1 :]
                cand = PlateText(region=pt.region, classification=new_cls, hiragana=pt.hiragana, serial=pt.serial)
                if not PlateValidator.validate(cand, plate_type, plate_type == PlateType.GRAPHIC):
                    candidates.append(cand)
        # serial mutations (one digit)
        s = str(pt.serial)
        for pos, ch in enumerate(s):
            for alt in _CONF_MAP.get(ch, []):
                if not alt.isdigit():
                    continue
                if pos == 0 and alt == "0":
                    continue
                new_s = s[:pos] + alt + s[pos + 1 :]
                if len(new_s) > 1 and new_s[0] == "0":
                    continue
                cand_serial = int(new_s)
                if not (1 <= cand_serial <= 9999):
                    continue
                if len(str(cand_serial)) != len(s):
                    continue
                cand = PlateText(region=pt.region, classification=pt.classification, hiragana=pt.hiragana, serial=cand_serial)
                if not PlateValidator.validate(cand, plate_type, plate_type == PlateType.GRAPHIC):
                    candidates.append(cand)
        if not candidates:
            return None
        return rng.choice(candidates)
