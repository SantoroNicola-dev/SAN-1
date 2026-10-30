"""
OCR Text Categorization & Guardrail Engine (Hybrid Fast-Path + Multilingual Semantic NLP)
=========================================================================================
Extracts and categorizes text inside images using RapidOCR (ONNX-powered)
and a 2-Tier Cascade Architecture:
- Tier 1: Microsecond Fast-Path using Aho-Corasick Automaton + Syntactic Pattern Matchers
          (blasphemy, profanity, identity documents, receipts/invoices).
- Tier 2: Multilingual Semantic NLP (ONNX MiniLM) for zero-shot categorization across
          inspirational quotes, commercial brands, greetings, hate speech, threats, and scams.
"""

from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
import re
import io
import unicodedata
import numpy as np
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPIDOCR_AVAILABLE = True
except ImportError:
    RAPIDOCR_AVAILABLE = False

from .aho_corasick import AhoCorasickAutomaton
from .text_semantic_guardrail import TextSemanticGuardrail


# ==============================================================================
# TIER 1: FAST-PATH LEXICONS & SYNTACTIC RULES
# ==============================================================================

ITALIAN_RELIGIOUS_SUBJECTS = {
    "dio", "madonna", "cristo", "gesu", "gesù", "padreterno", "onnipotente", "maria"
}

ITALIAN_BLASPHEMOUS_EPITHETS = {
    "porco", "porca", "cane", "cagna", "maiale", "maiala", "troia", "puttana",
    "bastardo", "bastarda", "boia", "ladro", "ladra", "infame", "merda", "schifoso",
    "schifosa", "lurido", "lurida", "vacca", "bestia", "verme", "fa schifo",
}

BLASPHEMY_SUBJECTS_REGEX = r"(?:dio|madonna|cristo|gesu|gesù|padreterno|onnipotente)"
BLASPHEMY_EPITHETS_REGEX = (
    r"(?:porco|porca|cane|cagna|maiale|maiala|troia|puttana|bastardo|bastarda|"
    r"boia|ladro|ladra|infame|merda|schifoso|schifosa|lurido|lurida|vacca|bestia|"
    r"verme|serpente|maledetto|maledetta|impiccato|faschifo|fa schifo)"
)

ITALIAN_BAD_WORDS = {
    "vaffanculo", "fanculo", "cazzo", "cazzi", "cazzone", "cazzata", "cazzate",
    "merda", "merde", "merdoso", "merdosa", "stronzo", "stronza", "stronzi", "stronze",
    "coglione", "coglioni", "cogliona", "troia", "troie", "puttana", "puttane", "puttaniere",
    "bastardo", "bastardi", "bastarda", "zoccola", "zoccole", "bagascia", "baldracca",
    "fottiti", "fottuto", "fottuta", "scopare", "scopata", "inculare", "inculato", "inculata",
    "chiavare", "pompinaro", "pompino", "pompini", "frocio", "froci", "finocchio",
    "negro", "negri", "negra", "leccaculo", "rompipalle", "rompicoglioni", "deficiente",
    "imbecille", "cretino", "cretina", "troietta", "mignotta", "cagna", "faccia da culo",
    "facciadaculo", "leccami il culo", "leccamiilculo", "tispezzoilculo", "spezzo il culo",
    "maiale schifoso", "maialeschifoso", "testa di cazzo", "testadicazzo", "pezzo di merda",
    "pezzodimerda", "figlio di puttana", "figliodiputtana", "minchia", "minchione", "troione",
    "bocchinara", "bocchinaro"
}

ENGLISH_UNCONDITIONAL_BAD_WORDS = {
    "fuck", "fucking", "fucked", "fucker", "fuckers", "fuckhead", "fuckface", "fuckboy", "fuckwit",
    "fucktard", "motherfucker", "motherfuckers", "motherfucking", "clusterfuck",
    "shit", "shitty", "bullshit", "shithole", "shithead", "shitfaced", "dipshit", "horseshit", "apeshit",
    "bitch", "bitches", "bitching", "bitchy", "bitchass",
    "asshole", "assholes", "dumbass", "dumbasses", "jackass", "jackasses", "fatass", "asswipe",
    "cunt", "cunts", "dickhead", "dickheads", "cocksucker", "cocksucking",
    "dick", "dicks", "pussy", "pussies", "crap", "prick", "pricks",
    "bastard", "bastards", "slut", "sluts", "slutty", "whore", "whores", "whoring",
    "twat", "twats", "wanker", "wankers",
    "retard", "retarded", "retards", "nigger", "niggers", "nigga", "niggas",
    "faggot", "faggots", "tranny", "trannies", "chink", "kike", "wetback",
    "scumbag", "scumbags", "douchebag", "douchebags",
    "fudk", "fudked", "fudkedhardo", "infidel", "infidels",
    "suicidebomb", "suicidebomber", "suicidebombing"
}

ENGLISH_RESTRICTED_BAD_WORDS = {
    "cock", "cocks",
    "ass", "asses", "tit", "tits", "boobs", "cum", "jizz", "blowjob", "handjob",
    "nazi", "hitler", "fuhrer", "rapist", "rapists", "pedophile", "pedo",
    "stfu", "gtfo", "kys", "wtf"
}

CORE_OFFENSIVE_PHRASES = {
    "shut the fuck up", "shut your fuck", "shut the hell up", "shut your mouth",
    "shut your whore mouth", "shut your filthy mouth", "shut your face",
    "go fuck yourself", "go to hell", "fuck off", "fuck you", "fuck that", "fuck this",
    "fuck this shit", "get the fuck out", "eat shit", "eat shit and die", "piece of shit",
    "suck my dick", "suck a dick", "suck my balls", "lick my balls", "lick my dick",
    "kiss my ass", "kill yourself", "die in a fire", "go die", "drop dead", "burn in hell",
    "fucking idiot", "fucking moron", "fucking loser", "fucking clown",
    "son of a bitch", "son of a whore", "piss off"
}

ENGLISH_BAD_WORDS = ENGLISH_UNCONDITIONAL_BAD_WORDS | ENGLISH_RESTRICTED_BAD_WORDS | CORE_OFFENSIVE_PHRASES

LEET_TABLE = str.maketrans({
    "@": "a", "4": "a",
    "8": "b",
    "3": "e",
    "1": "i", "!": "i", "|": "i",
    "0": "o",
    "5": "s", "$": "s",
    "7": "t", "+": "t",
    "*": "", "_": "", "-": ""
})


class OCRGuardrailResult:
    """Detailed text extraction and categorization outcome from the OCR Engine."""

    def __init__(
        self,
        is_allowed: bool,
        status: str,
        detected_text: str,
        category: str,
        rejection_code: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        offending_words: Optional[list[str]] = None,
        ocr_confidence: float = 0.0,
        semantic_scores: Optional[Dict[str, float]] = None,
        categories: Optional[List[Dict[str, Any]]] = None,
    ):
        self.is_allowed = is_allowed
        self.status = status
        self.detected_text = detected_text
        self.category = category
        self.rejection_code = rejection_code
        self.rejection_reason = rejection_reason
        self.offending_words = offending_words or []
        self.ocr_confidence = ocr_confidence
        self.semantic_scores = semantic_scores or {}
        self.categories = categories or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "detected_text": self.detected_text,
            "category": self.category,
            "ocr_confidence": round(self.ocr_confidence, 3),
            "categories": self.categories,
            "matched_keywords": self.offending_words,
            "semantic_scores": self.semantic_scores,
        }

    def __repr__(self):
        return f"<OCRGuardrailResult [{self.category}] text={self.detected_text[:30]!r} cats={len(self.categories)}>"


class OCRTextGuardrail:
    """
    State-of-the-Art OCR Text Extraction & Multi-Label Text Categorization Engine.
    """

    def __init__(self, use_cls: bool = False, enable_semantic: bool = True):
        if RAPIDOCR_AVAILABLE:
            self.engine = RapidOCR(use_cls=use_cls)
            if self.engine is not None and hasattr(self.engine, "text_det"):
                self.engine.text_det.limit_type = "max"
                self.engine.text_det.limit_side_len = 520
        else:
            self.engine = None

        self._init_fast_path_automata()

        self.enable_semantic = enable_semantic
        if self.enable_semantic:
            self.semantic_guard = TextSemanticGuardrail()
        else:
            self.semantic_guard = None

    def _init_fast_path_automata(self):
        """Compiles Aho-Corasick automata for microsecond multi-pattern matching."""
        unconditional_patterns = list(ITALIAN_BAD_WORDS | ENGLISH_UNCONDITIONAL_BAD_WORDS | CORE_OFFENSIVE_PHRASES)
        self.unconditional_ac = AhoCorasickAutomaton(unconditional_patterns)

        restricted_patterns = list(ENGLISH_RESTRICTED_BAD_WORDS)
        self.restricted_ac = AhoCorasickAutomaton(restricted_patterns)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalizes case, accents, and leetspeak characters."""
        if not text:
            return ""
        t = text.lower()
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")

        def _leet_token(match: re.Match) -> str:
            tok = match.group(0)
            if not re.search(r"[a-z]", tok):
                return tok
            return tok.translate(LEET_TABLE)

        t = re.sub(r"[a-z0-9@$!|*+_-]+", _leet_token, t)
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def categorize_text(
        self,
        raw_text: str,
        is_embedded_in_image: bool = False,
        ocr_confidence: float = 1.0,
        min_confidence_pct: float = 25.0,
        top_k: int = 5,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Categorizes raw or OCR-extracted text into weighted multi-label categories
        with independent confidence_pct (0-100%) and normalized share_pct (100% sum).
        Returns: (categories_list, matched_keywords)
        """
        if not raw_text or not raw_text.strip():
            return [], []

        norm = self.normalize_text(raw_text)
        collapsed = norm.replace(" ", "")
        if not norm:
            return [], []

        cat_map: Dict[str, Dict[str, Any]] = {}
        matched_keywords: List[str] = []

        def _add_cat(cid: str, label: str, group: str, conf: float, source: str = "text_lexical"):
            if cid not in cat_map or conf > cat_map[cid]["confidence_pct"]:
                cat_map[cid] = {
                    "id": cid,
                    "label": label,
                    "group": group,
                    "confidence_pct": round(float(np.clip(conf, 0.0, 99.9)), 2),
                    "source": source,
                }

        # 0. Embedded text overlay indicator when coming from image OCR
        if is_embedded_in_image and len(norm) >= 2:
            overlay_conf = min(96.0, max(45.0, float(ocr_confidence) * 95.0))
            _add_cat("text_overlay_caption", "Embedded Text / Graphic Caption", "text_content", overlay_conf, "ocr")

        # 1. Italian Religious Blasphemy check
        compound_match = re.search(
            rf"({BLASPHEMY_SUBJECTS_REGEX}{BLASPHEMY_EPITHETS_REGEX}|{BLASPHEMY_EPITHETS_REGEX}{BLASPHEMY_SUBJECTS_REGEX})",
            collapsed,
        )
        subj_match = re.search(rf"\b{BLASPHEMY_SUBJECTS_REGEX}\b", norm)
        ep_match = re.search(rf"\b{BLASPHEMY_EPITHETS_REGEX}", norm)
        if compound_match or (subj_match and ep_match):
            matched_str = compound_match.group(0) if compound_match else f"{ep_match.group(0)} {subj_match.group(0)}"
            matched_keywords.append(matched_str)
            _add_cat("blasphemy_religious_slur", "Religious Blasphemy (Bestemmia)", "text_content", 98.5, "text_lexical")

        # 2. Identity Document / Driver License PII check
        if re.search(
            r"\bdriver\s*license\b|\bdriverlicense\b|\bdriving\s*licen[sc]e\b|\bpatente\s*di\s*guida\b|\bcarta\s*d\s*identita\b|\bpassport\b|\bpassaporto\b",
            norm,
        ):
            matched_keywords.append("identity_document")
            _add_cat("identity_card_license", "Identity Card, Passport or Driver License", "documents_media", 96.0, "text_lexical")

        # 3. Receipt / Invoice lexical check
        if re.search(
            r"\b(?:receipt|invoice|scontrino|fattura|totale\s*euro|subtotal|cash\s*register|iva\s*inclusa)\b",
            norm,
        ):
            _add_cat("receipt_invoice", "Receipt, Invoice or Financial Bill", "documents_media", 90.0, "text_lexical")

        # 4. Profanity / Vulgar Words (Aho-Corasick)
        found_bad_words = set()
        for _, _, kw in self.unconditional_ac.search(norm, word_boundary=True):
            found_bad_words.add(kw)
        for _, _, kw in self.unconditional_ac.search(collapsed, word_boundary=False):
            if len(kw) >= 5:
                found_bad_words.add(kw)
        for _, _, kw in self.restricted_ac.search(norm, word_boundary=True):
            found_bad_words.add(kw)

        if found_bad_words:
            bad_list = sorted(list(found_bad_words))
            matched_keywords.extend(bad_list)
            conf_prof = min(98.0, 86.0 + 4.0 * len(bad_list))
            _add_cat("profanity_vulgar_text", "Profanity, Swearing or Vulgar Quote", "text_content", conf_prof, "text_lexical")

        # 5. Tier-2 Multilingual Semantic NLP Categories
        words_count = len(norm.split())
        if self.enable_semantic and self.semantic_guard and self.semantic_guard.is_ready and (words_count >= 2 or len(norm) >= 6):
            sem_cats = self.semantic_guard.classify_text(
                norm,
                min_confidence_pct=min_confidence_pct,
                top_k=top_k,
            )
            for sc in sem_cats:
                _add_cat(sc["id"], sc["label"], sc["group"], sc["confidence_pct"], sc["source"])

        # Filter and compute normalized share_pct
        sorted_cats = sorted(cat_map.values(), key=lambda x: x["confidence_pct"], reverse=True)
        active = [c for c in sorted_cats if c["confidence_pct"] >= min_confidence_pct]
        if not active and sorted_cats:
            active = [sorted_cats[0]]
        if top_k and top_k > 0:
            active = active[:top_k]

        total_conf = sum(c["confidence_pct"] for c in active) or 1.0
        running_share = 0.0
        for i, c in enumerate(active):
            if i == len(active) - 1:
                share = round(max(0.0, 100.0 - running_share), 2)
            else:
                share = round((c["confidence_pct"] / total_conf) * 100.0, 2)
                running_share += share
            c["share_pct"] = share

        return active, matched_keywords

    def check_text_safety(
        self, raw_text: str
    ) -> tuple[bool, str, Optional[str], Optional[str], list[str], Dict[str, float]]:
        """
        Two-Tier Cascade Validation (maintained for backward compatibility).
        """
        if not raw_text or not raw_text.strip():
            return True, "NO_TEXT_DETECTED", None, None, [], {}

        norm = self.normalize_text(raw_text)
        collapsed = norm.replace(" ", "")

        # 1. Blasphemy
        compound_match = re.search(
            rf"({BLASPHEMY_SUBJECTS_REGEX}{BLASPHEMY_EPITHETS_REGEX}|{BLASPHEMY_EPITHETS_REGEX}{BLASPHEMY_SUBJECTS_REGEX})",
            collapsed,
        )
        if compound_match:
            matched_str = compound_match.group(0)
            return (
                False,
                "blasphemy_religious_slur",
                "BLASPHEMY_DETECTED",
                f"Detected religious blasphemy expression ({matched_str}).",
                [matched_str],
                {},
            )

        subj_match = re.search(rf"\b{BLASPHEMY_SUBJECTS_REGEX}\b", norm)
        ep_match = re.search(rf"\b{BLASPHEMY_EPITHETS_REGEX}", norm)
        if subj_match and ep_match:
            matched_phrase = f"{ep_match.group(0)} {subj_match.group(0)}"
            return (
                False,
                "blasphemy_religious_slur",
                "BLASPHEMY_DETECTED",
                f"Detected religious blasphemy expression ({matched_phrase}).",
                [matched_phrase],
                {},
            )

        # 2. Identity Document
        if re.search(
            r"\bdriver\s*license\b|\bdriverlicense\b|\bdriving\s*licen[sc]e\b|\bpatente\s*di\s*guida\b|\bcarta\s*d\s*identita\b",
            norm,
        ):
            return (
                False,
                "identity_card_license",
                "DOCUMENT_OR_SCREENSHOT",
                "Detected text corresponding to an identity document or driver license.",
                ["driver license PII"],
                {},
            )

        # 3. Profanity
        found_bad_words = set()
        for _, _, kw in self.unconditional_ac.search(norm, word_boundary=True):
            found_bad_words.add(kw)
        for _, _, kw in self.unconditional_ac.search(collapsed, word_boundary=False):
            if len(kw) >= 5:
                found_bad_words.add(kw)
        for _, _, kw in self.restricted_ac.search(norm, word_boundary=True):
            found_bad_words.add(kw)

        if found_bad_words:
            bad_list = sorted(list(found_bad_words))
            cat = "profanity_vulgar_text"
            code = "PROFANITY_DETECTED"
            reason = f"Detected profanity or vulgar expressions ({', '.join(bad_list[:2])})."
            return False, cat, code, reason, bad_list, {}

        # 4. Semantic NLP
        words_count = len(norm.split())
        if self.enable_semantic and self.semantic_guard and self.semantic_guard.is_ready and (words_count >= 2 or len(norm) >= 8):
            sem_allowed, sem_cat, tox_sim, ben_sim, sem_reason = self.semantic_guard.evaluate_text(norm)
            sem_scores = {
                "toxic_similarity": round(tox_sim, 3),
                "benign_similarity": round(ben_sim, 3),
                "margin": round(tox_sim - ben_sim, 3),
            }
            if not sem_allowed:
                return (
                    False,
                    sem_cat,
                    "SENSITIVE_TEXT_DETECTED",
                    sem_reason or "Detected sensitive semantic text category.",
                    [sem_cat.replace("BLOCKED_SEMANTIC_", "")],
                    sem_scores,
                )

        return True, "text_overlay_caption", None, None, [], {}

    def validate_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image, np.ndarray],
        min_confidence_pct: float = 25.0,
        top_k: int = 5,
    ) -> OCRGuardrailResult:
        """
        Runs RapidOCR on the input image, extracts embedded text, and computes
        multi-label text categories with percentages.
        """
        if self.engine is None:
            return OCRGuardrailResult(
                is_allowed=True,
                status="ANALYZED",
                detected_text="",
                category="OCR_UNAVAILABLE",
                categories=[],
            )

        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(str(image_input)).convert("RGB")
        elif isinstance(image_input, bytes):
            pil_img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        max_dim = 960
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)

        np_bgr = np.array(pil_img)[:, :, ::-1]

        try:
            ocr_results, _ = self.engine(np_bgr)
        except Exception:
            return OCRGuardrailResult(
                is_allowed=True,
                status="ANALYZED",
                detected_text="",
                category="OCR_ERROR_FALLBACK",
                categories=[],
            )

        if not ocr_results:
            return OCRGuardrailResult(
                is_allowed=True,
                status="ANALYZED",
                detected_text="",
                category="NO_TEXT_DETECTED",
                categories=[],
            )

        text_lines = [item[1] for item in ocr_results if len(item) > 1 and item[1]]
        full_text = " ".join(text_lines)
        avg_conf = float(np.mean([item[2] for item in ocr_results if len(item) > 2])) if ocr_results else 1.0

        categories, matched_kw = self.categorize_text(
            full_text,
            is_embedded_in_image=True,
            ocr_confidence=avg_conf,
            min_confidence_pct=min_confidence_pct,
            top_k=top_k,
        )
        is_allowed, legacy_cat, code, reason, offending, sem_scores = self.check_text_safety(full_text)
        primary_cat = categories[0]["id"] if categories else legacy_cat

        return OCRGuardrailResult(
            is_allowed=is_allowed,
            status="ANALYZED",
            detected_text=full_text,
            category=primary_cat,
            rejection_code=code,
            rejection_reason=reason,
            offending_words=matched_kw or offending,
            ocr_confidence=avg_conf,
            semantic_scores=sem_scores,
            categories=categories,
        )

    def classify_and_route(
        self,
        image_path: Union[str, Path],
        move: bool = True,
        target_folder_name: Optional[str] = None,
    ) -> tuple[OCRGuardrailResult, Path]:
        """
        Categorizes the image text and optionally moves it into `<parent_dir>/<primary_category>/<filename>`.
        """
        p = Path(image_path)
        res = self.validate_image(p)
        if not move or not p.exists():
            return res, p

        subfolder = target_folder_name or (res.categories[0]["id"] if res.categories else "no_text")
        dest_dir = p.parent / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / p.name

        if p.resolve() != dest_path.resolve():
            p.replace(dest_path)

        return res, dest_path
