"""
OCR Text Guardrail Engine (Hybrid Fast-Path + Multilingual Semantic NLP)
========================================================================
Detects and filters text inside images using RapidOCR (ONNX-powered)
and a 2-Tier Cascade Guardrail Architecture:
- Tier 1: Microsecond Fast-Path using Aho-Corasick Automaton + Syntactic Blasphemy Matcher.
- Tier 2: Multilingual Semantic NLP Guardrail (ONNX) using contrastive policy embeddings
          to catch threats, hate speech, terrorism, self-harm, and harassment without manual lists.
- ALLOWS: Harmless, positive, neutral, or branding text (scritte innocue, marchi, ricette).
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
# TIER 1: FAST-PATH LEXICONS & SYNTACTIC BLASPHEMY RULES
# ==============================================================================

# 1. Italian Religious Blasphemies ("Bestemmie / Blasfemie")
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

# 2. Italian Core Bad Words & Slurs
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

# 3. English Unconditional Bad Words & Slurs
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

# 4. English Restricted Words (require strict word boundary to prevent false positives)
ENGLISH_RESTRICTED_BAD_WORDS = {
    "cock", "cocks",
    "ass", "asses", "tit", "tits", "boobs", "cum", "jizz", "blowjob", "handjob",
    "nazi", "hitler", "fuhrer", "rapist", "rapists", "pedophile", "pedo",
    "stfu", "gtfo", "kys", "wtf"
}

# 5. Core Vulgar Phrases (for instant fast-path match)
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

# Backward compatibility union
ENGLISH_BAD_WORDS = ENGLISH_UNCONDITIONAL_BAD_WORDS | ENGLISH_RESTRICTED_BAD_WORDS | CORE_OFFENSIVE_PHRASES

# Leetspeak translation table
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
    """Detailed validation outcome from the OCR Text Guardrail."""

    def __init__(
        self,
        is_allowed: bool,
        status: str,  # "APPROVED" or "BLOCKED"
        detected_text: str,
        category: str,
        rejection_code: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        offending_words: Optional[list[str]] = None,
        ocr_confidence: float = 0.0,
        semantic_scores: Optional[Dict[str, float]] = None,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_allowed": self.is_allowed,
            "status": self.status,
            "detected_text": self.detected_text,
            "category": self.category,
            "rejection_code": self.rejection_code,
            "rejection_reason": self.rejection_reason,
            "offending_words": self.offending_words,
            "ocr_confidence": round(self.ocr_confidence, 3),
            "semantic_scores": self.semantic_scores,
        }

    def __repr__(self):
        return f"<OCRGuardrailResult {self.status} [{self.category}] words={self.offending_words}>"


class OCRTextGuardrail:
    """
    State-of-the-Art OCR Text Extraction & Content Moderation Guardrail.
    Combines:
    - RapidOCR (ONNX) for sub-50ms text detection and recognition.
    - Tier 1: Microsecond Fast-Path (Aho-Corasick + Syntactic Blasphemy + Leetspeak normalizer).
    - Tier 2: Multilingual Semantic NLP (ONNX MiniLM) for zero-shot hate speech,
              violence, suicide, and harassment detection without manual wordlists.
    """

    def __init__(self, use_cls: bool = False, enable_semantic: bool = True):
        # 1. OCR Engine
        if RAPIDOCR_AVAILABLE:
            self.engine = RapidOCR(use_cls=use_cls)
            if self.engine is not None and hasattr(self.engine, "text_det"):
                self.engine.text_det.limit_type = "max"
                self.engine.text_det.limit_side_len = 520
        else:
            self.engine = None
            print("Warning: rapidocr_onnxruntime is not installed. OCR will be disabled.")

        # 2. Tier 1 Fast-Path Automata
        self._init_fast_path_automata()

        # 3. Tier 2 Semantic NLP Guardrail
        self.enable_semantic = enable_semantic
        if self.enable_semantic:
            self.semantic_guard = TextSemanticGuardrail()
        else:
            self.semantic_guard = None

    def _init_fast_path_automata(self):
        """Compiles Aho-Corasick automata for microsecond multi-pattern matching."""
        # Unconditional words & core phrases (can match anywhere or bounded)
        unconditional_patterns = list(ITALIAN_BAD_WORDS | ENGLISH_UNCONDITIONAL_BAD_WORDS | CORE_OFFENSIVE_PHRASES)
        self.unconditional_ac = AhoCorasickAutomaton(unconditional_patterns)

        # Restricted short words (must be bounded by word boundaries)
        restricted_patterns = list(ENGLISH_RESTRICTED_BAD_WORDS)
        self.restricted_ac = AhoCorasickAutomaton(restricted_patterns)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalizes case, accents, and leetspeak characters."""
        if not text:
            return ""
        # 1. Lowercase
        t = text.lower()
        # 2. Decompose accents
        t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
        # 3. Leetspeak mapping (preserving pure numbers like 717, dates, prices)
        def _leet_token(match: re.Match) -> str:
            tok = match.group(0)
            if not re.search(r"[a-z]", tok):
                return tok
            return tok.translate(LEET_TABLE)

        t = re.sub(r"[a-z0-9@$!|*+_-]+", _leet_token, t)
        # 4. Clean non-alphanumeric except spaces
        t = re.sub(r"[^a-z0-9\s]", " ", t)
        # 5. Compress multiple spaces
        t = re.sub(r"\s+", " ", t).strip()
        return t

    def check_text_safety(self, raw_text: str) -> tuple[bool, str, Optional[str], Optional[str], list[str], Dict[str, float]]:
        """
        Two-Tier Cascade Validation:
        - Tier 1: Fast-Path (<0.2ms) Blasphemy, Aho-Corasick Bad Words, Document PII.
        - Tier 2: Multilingual Semantic NLP (~8ms) for Hate Speech, Violence, Suicide, Threats.
        Returns: (is_allowed, category, rejection_code, rejection_reason, offending_words, semantic_scores)
        """
        if not raw_text or not raw_text.strip():
            return True, "NO_TEXT_DETECTED", None, None, [], {}

        norm = self.normalize_text(raw_text)
        collapsed = norm.replace(" ", "")

        # ======================================================================
        # TIER 1: FAST-PATH DETERMINISTIC SHORT-CIRCUITS (<0.2 ms)
        # ======================================================================

        # 1. Italian Religious Blasphemy / Bestemmie Syntactic Check
        compound_match = re.search(
            rf"({BLASPHEMY_SUBJECTS_REGEX}{BLASPHEMY_EPITHETS_REGEX}|{BLASPHEMY_EPITHETS_REGEX}{BLASPHEMY_SUBJECTS_REGEX})",
            collapsed,
        )
        if compound_match:
            matched_str = compound_match.group(0)
            return (
                False,
                "BLOCKED_BLASPHEMY",
                "BLASPHEMY_PROHIBITED",
                f"The image contains prohibited blasphemous expressions or religious slurs ({matched_str}).",
                [matched_str],
                {},
            )

        subj_match = re.search(rf"\b{BLASPHEMY_SUBJECTS_REGEX}\b", norm)
        ep_match = re.search(rf"\b{BLASPHEMY_EPITHETS_REGEX}", norm)
        if subj_match and ep_match:
            matched_phrase = f"{ep_match.group(0)} {subj_match.group(0)}"
            return (
                False,
                "BLOCKED_BLASPHEMY",
                "BLASPHEMY_PROHIBITED",
                f"The image contains prohibited blasphemous expressions or religious slurs ({matched_phrase}).",
                [matched_phrase],
                {},
            )

        # 2. Document PII / Identity Documents Check
        if re.search(r"\bdriver\s*license\b|\bdriverlicense\b|\bdriving\s*licen[sc]e\b|\bpatente\s*di\s*guida\b|\bcarta\s*d\s*identita\b", norm):
            return (
                False,
                "BLOCKED_DOCUMENT",
                "DOCUMENT_OR_SCREENSHOT",
                "The image contains text resembling personal identity documents or a driver license.",
                ["driver license PII"],
                {},
            )

        # 3. Fast-Path Aho-Corasick Matching (Unconditional Bad Words & Phrases)
        found_bad_words = set()

        # Check in normalized spaced text (word bounded)
        for _, _, kw in self.unconditional_ac.search(norm, word_boundary=True):
            found_bad_words.add(kw)

        # Check in collapsed text for words >= 5 characters (catches space-split elusions like 'f u c k')
        for _, _, kw in self.unconditional_ac.search(collapsed, word_boundary=False):
            if len(kw) >= 5:
                found_bad_words.add(kw)

        # Check restricted short words with strict boundary in normalized text
        for _, _, kw in self.restricted_ac.search(norm, word_boundary=True):
            found_bad_words.add(kw)

        if found_bad_words:
            bad_list = sorted(list(found_bad_words))
            cat = "BLOCKED_PROFANE_TEXT" if any(w in ITALIAN_BAD_WORDS for w in bad_list) else "BLOCKED_VULGAR_QUOTE"
            code = "BAD_WORD_DETECTED" if cat == "BLOCKED_PROFANE_TEXT" else "OFFENSIVE_QUOTE_DETECTED"
            reason = f"The image contains prohibited profanity, slurs, or vulgar text ({', '.join(bad_list[:2])})."
            return False, cat, code, reason, bad_list, {}

        # ======================================================================
        # TIER 2: MULTILINGUAL SEMANTIC NLP EVALUATION (ONNX)
        # ======================================================================
        # Invoked for texts with substantive linguistic content (>= 2 words or >= 8 chars)
        words_count = len(norm.split())
        if self.enable_semantic and self.semantic_guard and self.semantic_guard.is_ready and (words_count >= 2 or len(norm) >= 8):
            sem_allowed, sem_cat, tox_sim, ben_sim, sem_reason = self.semantic_guard.evaluate_text(norm)
            sem_scores = {"toxic_similarity": round(tox_sim, 3), "benign_similarity": round(ben_sim, 3), "margin": round(tox_sim - ben_sim, 3)}

            if not sem_allowed:
                return (
                    False,
                    sem_cat,
                    "TOXIC_CONTENT_DETECTED",
                    sem_reason or "The image contains prohibited hate speech, threats, or harassment.",
                    [sem_cat.replace("BLOCKED_SEMANTIC_", "")],
                    sem_scores,
                )

        # ======================================================================
        # ALLOWED: INNOCUOUS TEXT (SAFE)
        # ======================================================================
        return True, "ALLOWED_INNOCUOUS_TEXT", None, None, [], {}

    def validate_image(self, image_input: Union[str, Path, bytes, Image.Image, np.ndarray]) -> OCRGuardrailResult:
        """
        Runs RapidOCR on the input image and evaluates all extracted text blocks
        using the Hybrid Fast-Path + Multilingual Semantic NLP engine.
        """
        if self.engine is None:
            return OCRGuardrailResult(
                is_allowed=True,
                status="APPROVED",
                detected_text="",
                category="OCR_UNAVAILABLE",
            )

        # Load image into numpy RGB array
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

        # Optimize inference: downsample large images to max 960px for high-throughput OCR
        max_dim = 960
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)

        # Convert RGB to BGR for RapidOCR (OpenCV/PaddleOCR expects BGR format)
        np_bgr = np.array(pil_img)[:, :, ::-1]

        # Run OCR
        try:
            ocr_results, _ = self.engine(np_bgr)
        except Exception:
            return OCRGuardrailResult(
                is_allowed=True,
                status="APPROVED",
                detected_text="",
                category="OCR_ERROR_FALLBACK",
            )

        if not ocr_results:
            return OCRGuardrailResult(
                is_allowed=True,
                status="APPROVED",
                detected_text="",
                category="NO_TEXT_DETECTED",
            )

        # Combine text segments
        text_lines = [item[1] for item in ocr_results if len(item) > 1 and item[1]]
        full_text = " ".join(text_lines)
        avg_conf = float(np.mean([item[2] for item in ocr_results if len(item) > 2])) if ocr_results else 1.0

        is_allowed, category, code, reason, offending, sem_scores = self.check_text_safety(full_text)

        return OCRGuardrailResult(
            is_allowed=is_allowed,
            status="APPROVED" if is_allowed else "BLOCKED",
            detected_text=full_text,
            category=category,
            rejection_code=code,
            rejection_reason=reason,
            offending_words=offending,
            ocr_confidence=avg_conf,
            semantic_scores=sem_scores,
        )

    def classify_and_route(
        self,
        image_path: Union[str, Path],
        move: bool = True,
        accept_folder_name: str = "accepted",
        deny_folder_name: str = "deny",
    ) -> tuple[OCRGuardrailResult, Path]:
        r"""
        Validates the image text and sorts/moves it into relative '\accepted' or '\deny' subfolder:
        - If ALLOWED (Safe): moves to <parent_dir>/accepted/<filename>
        - If BLOCKED (Unsafe): moves to <parent_dir>/deny/<filename>
        """
        p = Path(image_path)
        res = self.validate_image(p)
        if not move or not p.exists():
            return res, p

        target_subfolder = accept_folder_name if res.is_allowed else deny_folder_name
        base_dir = p.parent.parent if p.parent.name.lower() in ("images", "image", "accept", "accepted", "deny") else p.parent
        dest_dir = base_dir / target_subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / p.name

        if p.resolve() != dest_path.resolve():
            p.replace(dest_path)

        return res, dest_path
