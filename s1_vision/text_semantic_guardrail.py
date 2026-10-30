"""
Multilingual Text Semantic Categorization & Moderation Engine (ONNX-powered)
============================================================================
Uses a quantized multilingual sentence transformer (Paraphrase-Multilingual-MiniLM-L12-v2)
to categorize text across general, commercial, motivational, and sensitive semantic categories
in Italian, English, and 50+ languages with calibrated multi-label percentages.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    NLP_ONNX_AVAILABLE = True
except ImportError:
    NLP_ONNX_AVAILABLE = False


DEFAULT_NLP_DIR = Path(__file__).resolve().parent.parent / "models" / "nlp"
DEFAULT_MODEL_PATH = DEFAULT_NLP_DIR / "model_quantized.onnx"
DEFAULT_TOKENIZER_PATH = DEFAULT_NLP_DIR / "tokenizer.json"


# Sensitive & Prohibited Policy Anchor Sentences
SEMANTIC_POLICY_ANCHORS = {
    "HATE_AND_DISCRIMINATION": [
        "hate speech against protected groups, racist slurs, ethnic discrimination, antisemitism, white supremacy, bigotry",
        "trans people should not exist, transphobic hate, homophobic slurs, hatred against minorities",
        "antisemitic holocaust remarks, jews in ovens, neo-nazi propaganda, ethnic cleansing",
        "odio razziale, insulti razzisti, discriminazione contro le minoranze, antisemitismo, suprematismo",
    ],
    "VIOLENCE_AND_THREATS": [
        "death threats, murder threat, I will kill you, kill your family, shoot you with a gun",
        "encouraging suicide, go kill yourself, tying a noose around neck, drop dead, burn in hell",
        "mass shootings, terrorism, suicide bombing, killing infidels, 9/11 terrorist attacks, bomb mosque",
        "minacce di morte, ti ammazzo, ti sparo, terrorismo, strage, attentato bomba, istigazione al suicidio",
    ],
    "CRUELTY_AND_ABUSE": [
        "grotesque cruelty, infanticide, killing babies, sadistic violence, child abuse",
        "rape and murder, sexual assault threats, violent physical torture, mutilation",
        "violenza sadica, uccidere neonati, stupro, tortura, crudeltà efferata",
    ],
    "HARASSMENT_AND_DEGRADATION": [
        "severe abusive insults, disgusting filth, worthless piece of trash, waste of oxygen, rot in hell",
        "vile personal harassment, extreme bullying, toxic humiliation, drop dead loser",
        "insulti degradanti gravissimi, pezzo di merda devi morire, feccia umana inutile",
    ],
    "CYBERATTACKS_AND_MALWARE": [
        "create a computer worm, develop ransomware, hacking bank accounts, write keylogger, malware exploit script, virus payload, trojan horse, botnet attack, steal passwords, cyber attack",
        "creare malware, programmare un ransomware, virus informatico, hackerare conti bancari, exploit di sicurezza, keylogger, attacco informatico, rubare credenziali",
    ],
    "FRAUD_AND_SCAMS": [
        "create a scam, online fraud operation, counterfeit money, fake passport and identity, credit card skimming, identity theft, ponzi pyramid scheme, tax evasion tricks, financial fraud",
        "creare una truffa, frode online, clonazione carte di credito, denaro contraffatto, documenti falsi, schema ponzi, evasione fiscale, truffe telematiche",
    ],
    "ILLEGAL_ACTS_AND_CONTRABAND": [
        "smuggling illegal drugs across borders, traffic contraband, laundering money, armed robbery heist, evade police detection, illegal gambling ring, shoplifting burglary, criminal enterprise",
        "traffico di droga, contrabbando, riciclaggio di denaro sporco, rapina a mano armata, eludere le forze dell'ordine, furto con scasso, attività criminali",
    ],
    "DANGEROUS_WEAPONS_AND_BOMBS": [
        "how to make a pipe bomb, synthesize chemical poison, manufacture homemade explosives, assemble untraceable ghost gun, detonator trigger mechanism, biological weapon",
        "costruire una bomba artigianale, sintetizzare veleno letale, fabbricare esplosivi, assemblare armi clandestine, detonatore, ordigni esplosivi",
    ],
    "SURVEILLANCE_AND_PRIVACY_VIOLATION": [
        "secretly record someone in private residence, install hidden spy camera, unauthorized surveillance, spy on someone, wiretapping, stalking, doxxing personal address, tracking individuals without consent",
        "registrare di nascosto in casa privata, telecamera spia nascosta, sorveglianza non autorizzata, spiare persone, intercettazioni, stalking, doxxing, tracciare persone senza consenso",
    ],
}

# General & Benign Semantic Categories
DESCRIPTIVE_TEXT_ANCHORS = {
    "greeting_social_caption": {
        "label": "Friendly Greeting or Social Caption",
        "group": "text_content",
        "phrases": [
            "friendly greetings, hello everyone, have a nice day, welcome, love, kindness, smile, peace",
            "scenic travel caption, beautiful sunny day, holiday vacation memories, family portrait photo",
            "saluti amichevoli, buongiorno a tutti, buona giornata, pace, amore, gentilezza, felicità",
        ],
    },
    "inspirational_quote": {
        "label": "Inspirational or Motivational Quote",
        "group": "text_content",
        "phrases": [
            "inspirational quote, motivational life philosophy, hard work, success, dreams, happiness, hope",
            "citazione motivazionale, credi nei tuoi sogni, impegno e lavoro, successo personale, saggezza",
        ],
    },
    "commercial_brand_logo": {
        "label": "Brand Logo, Label or Product Text",
        "group": "text_content",
        "phrases": [
            "fashion apparel clothing brand, casual t-shirt graphic logo, athletic sportswear brand, merchandise",
            "harmless product label, 100% cotton made in Italy, washing instructions, food recipe, grocery item",
            "marchio di abbigliamento, etichetta tessuto, cotone, made in italy, ricetta, indicazioni stradali",
        ],
    },
    "humorous_joke_text": {
        "label": "Humorous Joke or Satirical Caption",
        "group": "text_content",
        "phrases": [
            "funny harmless meme, humorous innocent joke, cute animal quote, playful friendly banter",
            "battuta divertente, scherzo simpatico, ironia leggera, meme umoristico",
        ],
    },
}

BENIGN_ANCHORS = [
    p
    for info in DESCRIPTIVE_TEXT_ANCHORS.values()
    for p in info["phrases"]
]

SENSITIVE_CATEGORY_MAPPING = {
    "HATE_AND_DISCRIMINATION": ("hate_speech_text", "Hate Speech & Discrimination (Text)"),
    "VIOLENCE_AND_THREATS": ("threats_violence_text", "Violence, Death Threats or Self-Harm (Text)"),
    "CRUELTY_AND_ABUSE": ("threats_violence_text", "Cruelty, Abuse or Graphic Violence (Text)"),
    "HARASSMENT_AND_DEGRADATION": ("profanity_vulgar_text", "Harassment, Insults or Degradation (Text)"),
    "CYBERATTACKS_AND_MALWARE": ("cyber_malware_text", "Cyberattack, Hacking or Malware (Text)"),
    "FRAUD_AND_SCAMS": ("scam_fraud_text", "Financial Scam or Fraud (Text)"),
    "ILLEGAL_ACTS_AND_CONTRABAND": ("scam_fraud_text", "Illegal Acts or Contraband (Text)"),
    "DANGEROUS_WEAPONS_AND_BOMBS": ("weapon_firearm_knife", "Dangerous Weapons or Explosives (Text)"),
    "SURVEILLANCE_AND_PRIVACY_VIOLATION": ("scam_fraud_text", "Surveillance or Privacy Violation (Text)"),
}


class TextSemanticGuardrail:
    """
    Tier-2 Semantic NLP Categorizer & Guardrail.
    Embeds input text into 384-dimensional latent space and computes both
    multi-label semantic categories with percentages and contrastive safety scores.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        tokenizer_path: Union[str, Path] = DEFAULT_TOKENIZER_PATH,
        margin_threshold: float = 0.05,
        min_toxic_threshold: float = 0.52,
        max_seq_length: int = 128,
    ):
        self.margin_threshold = margin_threshold
        self.min_toxic_threshold = min_toxic_threshold
        self.max_seq_length = max_seq_length
        self.is_ready = False
        self.session = None
        self.tokenizer = None

        m_path = Path(model_path)
        t_path = Path(tokenizer_path)

        if not NLP_ONNX_AVAILABLE:
            return

        if not m_path.exists() or not t_path.exists():
            return

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(m_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.tokenizer = Tokenizer.from_file(str(t_path))

            self._init_anchors()
            self.is_ready = True
        except Exception as e:
            print(f"[TextSemanticGuardrail] Initialization failed: {e}")
            self.is_ready = False

    def _init_anchors(self):
        """Encodes and pre-caches normalized policy and descriptive vectors."""
        self.toxic_cat_vectors = {}
        all_toxic_vecs = []
        all_toxic_cats = []

        for cat_name, phrases in SEMANTIC_POLICY_ANCHORS.items():
            vecs = self.encode(phrases)
            self.toxic_cat_vectors[cat_name] = vecs
            for v in vecs:
                all_toxic_vecs.append(v)
                all_toxic_cats.append(cat_name)

        self.all_toxic_matrix = np.array(all_toxic_vecs)
        self.all_toxic_cats = all_toxic_cats

        # Descriptive categories
        self.descriptive_cat_vectors = {}
        for cat_id, info in DESCRIPTIVE_TEXT_ANCHORS.items():
            self.descriptive_cat_vectors[cat_id] = self.encode(info["phrases"])

        self.benign_matrix = self.encode(BENIGN_ANCHORS)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Tokenizes and embeds text batch, applying mean-pooling and L2 normalization.
        Returns: (N, 384) float32 numpy array.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        encoded = [self.tokenizer.encode(t) for t in texts]
        batch_size = len(texts)
        max_len = min(max(len(e.ids) for e in encoded), self.max_seq_length)
        max_len = max(max_len, 1)

        input_ids = np.zeros((batch_size, max_len), dtype=np.int64)
        attention_mask = np.zeros((batch_size, max_len), dtype=np.int64)
        token_type_ids = np.zeros((batch_size, max_len), dtype=np.int64)

        for i, e in enumerate(encoded):
            ids = e.ids[:max_len]
            att = e.attention_mask[:max_len]
            input_ids[i, :len(ids)] = ids
            attention_mask[i, :len(att)] = att

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        outputs = self.session.run(None, feed)
        last_hidden_state = outputs[0]

        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_emb = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_emb / sum_mask

        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norms, a_min=1e-9, a_max=None)

    def classify_text(
        self,
        text: str,
        min_confidence_pct: float = 25.0,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Categorizes arbitrary text into descriptive and sensitive semantic categories,
        returning independent confidence_pct (0-100%) and normalized share_pct (100% sum).
        """
        if not self.is_ready or not text or len(text.strip()) < 3:
            return []

        v = self.encode([text])[0]
        raw_candidates: Dict[str, Dict[str, Any]] = {}

        # 1. Score descriptive categories
        for cat_id, vecs in self.descriptive_cat_vectors.items():
            sim = float(np.max(np.dot(vecs, v)))
            info = DESCRIPTIVE_TEXT_ANCHORS[cat_id]
            logit = 11.0 * (sim - 0.28)
            conf = 100.0 / (1.0 + float(np.exp(-np.clip(logit, -15.0, 15.0))))
            raw_candidates[cat_id] = {
                "id": cat_id,
                "label": info["label"],
                "group": info["group"],
                "similarity": round(sim, 4),
                "confidence_pct": round(float(np.clip(conf, 0.0, 99.9)), 2),
                "source": "nlp_semantic",
            }

        # 2. Score sensitive categories
        max_benign_sim = float(np.max(np.dot(self.benign_matrix, v)))
        for raw_cat, vecs in self.toxic_cat_vectors.items():
            sim = float(np.max(np.dot(vecs, v)))
            cat_id, cat_label = SENSITIVE_CATEGORY_MAPPING[raw_cat]
            margin = sim - max_benign_sim
            # Require meaningful elevation or positive margin against benign anchors
            if sim < 0.35 or (sim < self.min_toxic_threshold and margin < 0.02):
                conf = float(np.clip(sim * 40.0, 0.0, 18.0))
            else:
                logit = 12.5 * (sim - 0.34) + 3.0 * margin
                conf = 100.0 / (1.0 + float(np.exp(-np.clip(logit, -15.0, 15.0))))
                conf = float(np.clip(conf, 0.0, 99.9))

            if cat_id not in raw_candidates or conf > raw_candidates[cat_id]["confidence_pct"]:
                raw_candidates[cat_id] = {
                    "id": cat_id,
                    "label": cat_label,
                    "group": "text_content",
                    "similarity": round(sim, 4),
                    "confidence_pct": round(conf, 2),
                    "source": "nlp_semantic",
                }

        sorted_items = sorted(raw_candidates.values(), key=lambda x: x["confidence_pct"], reverse=True)
        active = [item for item in sorted_items if item["confidence_pct"] >= min_confidence_pct]
        if not active and sorted_items:
            active = [sorted_items[0]]
        if top_k and top_k > 0:
            active = active[:top_k]

        total_conf = sum(item["confidence_pct"] for item in active) or 1.0
        running_share = 0.0
        for i, item in enumerate(active):
            if i == len(active) - 1:
                share = round(max(0.0, 100.0 - running_share), 2)
            else:
                share = round((item["confidence_pct"] / total_conf) * 100.0, 2)
                running_share += share
            item["share_pct"] = share

        return active

    def evaluate_text(
        self,
        text: str,
    ) -> Tuple[bool, str, float, float, Optional[str]]:
        """
        Evaluates input text against semantic policy anchors (backward compatible).
        Returns:
            (is_allowed, category, max_toxic_sim, max_benign_sim, violation_reason)
        """
        if not self.is_ready or not text or len(text.strip()) < 4:
            return True, "ALLOWED_INNOCUOUS_TEXT", 0.0, 0.0, None

        v = self.encode([text])[0]
        toxic_sims = np.dot(self.all_toxic_matrix, v)
        benign_sims = np.dot(self.benign_matrix, v)

        max_toxic_idx = int(np.argmax(toxic_sims))
        max_toxic_sim = float(toxic_sims[max_toxic_idx])
        max_benign_sim = float(np.max(benign_sims))

        margin = max_toxic_sim - max_benign_sim
        matched_cat = self.all_toxic_cats[max_toxic_idx]

        is_blocked = (max_toxic_sim >= self.min_toxic_threshold) or (
            max_toxic_sim >= 0.38 and margin > self.margin_threshold
        )

        if is_blocked:
            reasons = {
                "HATE_AND_DISCRIMINATION": "The content contains hate speech, discrimination, or extremist ideology.",
                "VIOLENCE_AND_THREATS": "The content contains death threats, violent incitement, terrorism, or self-harm encouragement.",
                "CRUELTY_AND_ABUSE": "The content refers to grotesque cruelty, severe abuse, or graphic violence.",
                "HARASSMENT_AND_DEGRADATION": "The content contains severe abusive insults, targeted harassment, or degradation.",
                "CYBERATTACKS_AND_MALWARE": "The content provides instructions or material related to malware, cyberattacks, ransomware, or hacking.",
                "FRAUD_AND_SCAMS": "The content relates to scams, financial fraud, counterfeiting, or illegal schemes.",
                "ILLEGAL_ACTS_AND_CONTRABAND": "The content promotes or instructs on criminal activities, contraband, or robberies.",
                "DANGEROUS_WEAPONS_AND_BOMBS": "The content instructs on manufacturing explosive devices, poisons, or illegal weapons.",
                "SURVEILLANCE_AND_PRIVACY_VIOLATION": "The content refers to unauthorized wiretapping, hidden cameras, stalking, or privacy violations.",
            }
            reason = reasons.get(matched_cat, "The content matches a sensitive semantic category.")
            return False, f"BLOCKED_SEMANTIC_{matched_cat}", max_toxic_sim, max_benign_sim, reason

        return True, "ALLOWED_INNOCUOUS_TEXT", max_toxic_sim, max_benign_sim, None
