"""
Multilingual Text Semantic Moderation Engine (ONNX-powered)
===========================================================
Uses a quantized multilingual sentence transformer (Paraphrase-Multilingual-MiniLM-L12-v2)
to evaluate semantic toxicity, hate speech, threats, and harassment in Italian, English,
and 50+ languages without requiring hardcoded phrase lists.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
import os
import time
import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
    NLP_ONNX_AVAILABLE = True
except ImportError:
    NLP_ONNX_AVAILABLE = False


# Default Model Paths in workspace
DEFAULT_NLP_DIR = Path(__file__).resolve().parent.parent / "models" / "nlp"
DEFAULT_MODEL_PATH = DEFAULT_NLP_DIR / "model_quantized.onnx"
DEFAULT_TOKENIZER_PATH = DEFAULT_NLP_DIR / "tokenizer.json"


# Policy Anchor Sentences for Semantic Contrastive Space
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

BENIGN_ANCHORS = [
    "friendly greetings, hello everyone, have a nice day, welcome, love, kindness, smile, peace",
    "inspirational quote, motivational life philosophy, hard work, success, dreams, happiness, hope",
    "fashion apparel clothing brand, casual t-shirt graphic logo, athletic sportswear brand, merchandise",
    "harmless product label, 100% cotton made in Italy, washing instructions, food recipe, grocery item",
    "scenic travel caption, beautiful sunny day, holiday vacation memories, family portrait photo",
    "funny harmless meme, humorous innocent joke, cute animal quote, playful friendly banter",
    "saluti amichevoli, buongiorno a tutti, buona giornata, pace, amore, gentilezza, felicità",
    "citazione motivazionale, credi nei tuoi sogni, impegno e lavoro, successo personale",
    "marchio di abbigliamento, etichetta tessuto, cotone, made in italy, ricetta, indicazioni stradali",
]


class TextSemanticGuardrail:
    """
    Tier-2 Semantic NLP Guardrail.
    Embeds input text into 384-dimensional latent space and checks contrastive margin
    against prohibited semantic concepts versus benign expressions.
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
            print("[TextSemanticGuardrail] onnxruntime or tokenizers not installed.")
            return

        if not m_path.exists() or not t_path.exists():
            print(f"[TextSemanticGuardrail] Model files not found at {m_path.parent}.")
            return

        try:
            # Configure ONNX session for high-throughput single-core or multi-core CPU inference
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.session = ort.InferenceSession(
                str(m_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"]
            )
            self.tokenizer = Tokenizer.from_file(str(t_path))

            # Pre-compute anchor embeddings
            self._init_anchors()
            self.is_ready = True
        except Exception as e:
            print(f"[TextSemanticGuardrail] Initialization failed: {e}")
            self.is_ready = False

    def _init_anchors(self):
        """Encodes and pre-caches normalized policy vectors for zero-shot contrastive arbitration."""
        # 1. Toxic anchors by category
        self.toxic_cat_vectors = {}
        all_toxic_vecs = []
        all_toxic_cats = []

        for cat_name, phrases in SEMANTIC_POLICY_ANCHORS.items():
            vecs = self.encode(phrases)
            self.toxic_cat_vectors[cat_name] = vecs
            for v in vecs:
                all_toxic_vecs.append(v)
                all_toxic_cats.append(cat_name)

        self.all_toxic_matrix = np.array(all_toxic_vecs)  # Shape: (N_toxic, 384)
        self.all_toxic_cats = all_toxic_cats

        # 2. Benign anchors
        self.benign_matrix = self.encode(BENIGN_ANCHORS)  # Shape: (N_benign, 384)

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Tokenizes and embeds text batch, applying mean-pooling and L2 normalization.
        Returns: (N, 384) float32 numpy array.
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        # Batch encode with fast rust tokenizer
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

        # Run ONNX inference
        outputs = self.session.run(None, feed)
        last_hidden_state = outputs[0]  # Shape: (batch_size, max_len, 384)

        # Mean pooling with attention mask
        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_emb = np.sum(last_hidden_state * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        pooled = sum_emb / sum_mask

        # L2 Normalize
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.clip(norms, a_min=1e-9, a_max=None)

    def evaluate_text(
        self,
        text: str,
    ) -> Tuple[bool, str, float, float, Optional[str]]:
        """
        Evaluates input text against semantic policy anchors.
        Returns:
            (is_allowed, category, max_toxic_sim, max_benign_sim, violation_reason)
        """
        if not self.is_ready or not text or len(text.strip()) < 4:
            return True, "ALLOWED_INNOCUOUS_TEXT", 0.0, 0.0, None

        # Encode input text
        v = self.encode([text])[0]  # Shape: (384,)

        # Dot-product similarity against toxic and benign matrices
        toxic_sims = np.dot(self.all_toxic_matrix, v)
        benign_sims = np.dot(self.benign_matrix, v)

        max_toxic_idx = int(np.argmax(toxic_sims))
        max_toxic_sim = float(toxic_sims[max_toxic_idx])
        max_benign_sim = float(np.max(benign_sims))

        margin = max_toxic_sim - max_benign_sim
        matched_cat = self.all_toxic_cats[max_toxic_idx]

        # Contrastive arbitration rule:
        # Require toxic similarity to be meaningfully elevated (>= 0.38) before margin comparison
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
            reason = reasons.get(matched_cat, "The content violates platform safety guidelines.")
            return False, f"BLOCKED_SEMANTIC_{matched_cat}", max_toxic_sim, max_benign_sim, reason

        return True, "ALLOWED_INNOCUOUS_TEXT", max_toxic_sim, max_benign_sim, None
