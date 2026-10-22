import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.advisory.records import guidance_for_disease

class GroundedAdvisoryRAG:
    """Serve checked advisory records; retain FAISS only for explicit research use."""

    def __init__(
        self,
        knowledge_dir: Path = Path("knowledge"),
        index_path: Path = Path("data/processed/faiss_index.bin"),
        metadata_path: Path = Path("data/processed/kb_chunks.json"),
        model_name: str = "all-MiniLM-L6-v2",
        min_confidence_threshold: float = 0.70,
        min_retrieval_similarity: float = 0.35
    ):
        self.knowledge_dir = knowledge_dir
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.min_confidence_threshold = min_confidence_threshold
        self.min_retrieval_similarity = min_retrieval_similarity

        # Inference now uses exact disease-matched checked records. Keep the
        # legacy FAISS index available for explicit rebuild/research, but don't
        # load its embedding model in the farmer app or CLI path.
        self.model_name = model_name
        self.encoder = None
        self.index = None
        self.chunks = []

        # Explicitly building/loading FAISS remains available via
        # load_or_build_index(); it's not needed for evidence-gated inference.

    def load_or_build_index(self, force_rebuild: bool = False):
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        if self.encoder is None:
            print(f"[RAG] Loading embedding model '{self.model_name}' on CPU...")
            self.encoder = SentenceTransformer(self.model_name, device="cpu")
        if not force_rebuild and self.index_path.exists() and self.metadata_path.exists():
            print(f"[RAG] Loading existing FAISS index from {self.index_path}...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
            print(f"[RAG] Loaded {len(self.chunks)} knowledge chunks from disk.")
            return

        print(f"[RAG] Ingesting documents exclusively from {self.knowledge_dir.resolve()}...")
        self.chunks = []

        doc_files = list(self.knowledge_dir.glob("*.txt")) + list(self.knowledge_dir.glob("*.pdf"))
        if not doc_files:
            raise FileNotFoundError(f"No knowledge documents (.txt, .pdf) found in {self.knowledge_dir}")

        for doc_path in doc_files:
            if doc_path.suffix.lower() == ".txt":
                self._ingest_txt(doc_path)
            elif doc_path.suffix.lower() == ".pdf":
                self._ingest_pdf(doc_path)

        print(f"[RAG] Total extracted chunks: {len(self.chunks)}")
        corpus_texts = [c["text"] for c in self.chunks]
        embeddings = self.encoder.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # Cosine similarity because embeddings are normalized
        self.index.add(embeddings)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, indent=2)

        print(f"[RAG] Built FAISS index with {self.index.ntotal} vectors and saved to {self.index_path}")

    def _ingest_txt(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split into logical sections by delimiter or section tags
        raw_sections = re.split(r"-{20,}|={20,}", content)
        for s_idx, sec in enumerate(raw_sections):
            clean_sec = sec.strip()
            if len(clean_sec) < 40:
                continue

            # Identify section header
            lines = clean_sec.splitlines()
            title = lines[0].strip() if lines else f"Section {s_idx+1}"

            self.chunks.append({
                "chunk_id": len(self.chunks),
                "source_file": path.name,
                "source_page": 1,
                "section_title": title,
                "text": clean_sec
            })

    def _ingest_pdf(self, path: Path):
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            clean_text = text.strip()
            if len(clean_text) < 40:
                continue

            # Split page into 500-character overlapping chunks if long
            self.chunks.append({
                "chunk_id": len(self.chunks),
                "source_file": path.name,
                "source_page": page_num,
                "section_title": f"Page {page_num}",
                "text": clean_text
            })

    def get_advisory(
        self,
        predicted_disease: str,
        confidence: float,
        severity_band: str = "Moderate",
        severity_pct: Optional[float] = None,
        top_k: int = 3
    ) -> Dict:
        """Returns treatment advisory with strict refusal guarantees.
        Zero generative hallucinations: only verbatim retrieved text is presented."""

        # 1. Prediction Confidence Guard
        if confidence < self.min_confidence_threshold:
            return {
                "status": "REFUSAL_LOW_CONFIDENCE",
                "message": (
                    f"Low confidence ({confidence*100:.1f}% < {self.min_confidence_threshold*100:.0f}% threshold): "
                    "consult the local TNAU KVK / agricultural officer. Chemical advice withheld to avoid misapplication."
                ),
                "predicted_disease": predicted_disease,
                "confidence": confidence,
                "severity_band": severity_band,
                "retrieved_passages": [],
                "advisory_markdown": (
                    f"### Diagnostic Advisory: Withheld\n\n"
                    f"> [!WARNING]\n"
                    f"> **Low confidence diagnosis ({confidence*100:.1f}%)** for `{predicted_disease}`.\n"
                    f"> Consult the local TNAU KVK / agricultural officer for mandatory physical field verification.\n"
                    f"> **Zero chemical or pesticide advice is provided for low-confidence detections.**"
                )
            }

        # Only the checked structured source records are permitted to support
        # user-facing recommendations. Legacy local knowledge files contain
        # unverified/mismatched source text and are not suitable evidence.
        records = guidance_for_disease(predicted_disease, reliable=True)
        if not records:
            return {
                "status": "REFUSAL_LOW_RETRIEVAL_SIMILARITY",
                "message": "No verified disease-specific cultural/IPM record is available; consult the local TNAU KVK / agricultural officer.",
                "predicted_disease": predicted_disease,
                "confidence": confidence,
                "severity_band": severity_band,
                "retrieved_passages": [],
                "citations": [],
                "provenance_status": "no_matching_verified_record",
                "advisory_markdown": (
                    "### Advisory: No verified matching source record\n\n"
                    "> No disease-specific guidance is provided. Consult a local TNAU KVK or agricultural officer.\n"
                    "> Chemical products and doses are withheld."
                ),
            }

        record = records[0]
        citation = f"[{record['source_title']}, {record['source_page']}]({record['source_url']})"
        advisory_md = (
            f"## Source-checked cultural/IPM guidance\n\n"
            f"**Possible model match:** `{predicted_disease}` ({confidence*100:.1f}%; not a confirmed diagnosis)\n\n"
            f"### Cultural and IPM guidance\n{record['guidance_text']}\n\n"
            "### Chemical products and doses\nWithheld: no current applicable official product-label evidence has been verified.\n\n"
            f"### Source\n{citation}\n"
        )
        return {
            "status": "SUCCESS",
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "severity_band": severity_band,
            "retrieved_passages": [record],
            "citations": [citation],
            "provenance_status": record["evidence_status"],
            "advisory_markdown": advisory_md,
        }

        # 2. Vector Retrieval Query
        query = f"{predicted_disease} {severity_band} severity management cultural IPM chemical control TNAU CIBRC"
        query_vec = self.encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True)

        scores, indices = self.index.search(query_vec, top_k)
        top_scores = scores[0].tolist()
        top_indices = indices[0].tolist()

        max_score = top_scores[0]
        # 3. Retrieval Similarity Guard
        if max_score < self.min_retrieval_similarity:
            return {
                "status": "REFUSAL_LOW_RETRIEVAL_SIMILARITY",
                "message": (
                    f"Low retrieval score ({max_score:.3f} < {self.min_retrieval_similarity:.2f}): "
                    "no validated TNAU/CIBRC protocol found for query. Consult the local TNAU KVK / agricultural officer."
                ),
                "predicted_disease": predicted_disease,
                "confidence": confidence,
                "severity_band": severity_band,
                "retrieved_passages": [],
                "advisory_markdown": (
                    f"### Advisory: Ingestion Match Not Found\n\n"
                    f"> [!NOTE]\n"
                    f"> Low confidence: consult the local TNAU KVK / agricultural officer.\n"
                    f"> No registered management protocol matched in current knowledge documents."
                )
            }

        # 4. Extract retrieved passages
        passages = []
        for rank, (score, idx) in enumerate(zip(top_scores, top_indices), start=1):
            chunk = self.chunks[idx]
            passages.append({
                "rank": rank,
                "similarity_score": round(score, 4),
                "source_file": chunk["source_file"],
                "source_page": chunk["source_page"],
                "section_title": chunk["section_title"],
                "verbatim_text": chunk["text"]
            })

        # Separate Cultural/IPM and Chemical Measures from retrieved text.
        # The local corpus has no verifiable label/leaflet URLs or dated label
        # snapshots. Never surface product/dose instructions until those are
        # checked against current official product labels.
        cultural_ipm_lines = []
        source_citations = set()

        for p in passages:
            source_citations.add(f"{p['source_file']} (Page {p['source_page']})")
            text = p["verbatim_text"]

            # Extract Cultural section
            if "Cultural & IPM Measures" in text:
                try:
                    cult_part = text.split("Cultural & IPM Measures (First Priority):")[1]
                    cult_part = cult_part.split("Chemical Measures")[0].strip()
                    cultural_ipm_lines.append(cult_part)
                except Exception:
                    pass


        # Format Structured Advisory strictly adhering to required order:
        # Order: 1. Cultural/IPM measures first, 2. Chemical options exactly as listed, 3. Disclaimer
        advisory_md = f"## Retrieved Advisory Text (local reference provenance not independently verified)\n\n"
        advisory_md += f"**Diagnosis**: `{predicted_disease}` (Confidence: {confidence*100:.1f}%)  \n"
        sev_info = f"**Assessed Severity**: `{severity_band}`"
        if severity_pct is not None:
            sev_info += f" ({severity_pct:.2f}% lesion area - Heuristic)"
        advisory_md += f"{sev_info}  \n"
        advisory_md += f"**Primary Knowledge Source**: {', '.join(sorted(source_citations))}\n\n"

        advisory_md += "### 1. Cultural & IPM Measures (First Priority)\n"
        if cultural_ipm_lines:
            advisory_md += cultural_ipm_lines[0] + "\n\n"
        else:
            advisory_md += (
                "No relevant cultural/IPM passage was found in the current source documents. "
                "Please consult a TNAU KVK or agricultural officer.\n\n"
            )

        advisory_md += "### 2. Chemical Management Options (Strictly Grounded)\n"
        if severity_band == "Low":
            advisory_md += (
                "> [!TIP]\n"
                "> **Severity is LOW (<5% lesion coverage).** Chemical intervention is not economically justified.\n"
                "> Prioritize cultural practices and monitor fields weekly before resorting to synthetic chemicals.\n\n"
            )
        advisory_md += (
            "Chemical products and doses are withheld: the local reference files do not include "
            "verifiable, current product label/leaflet evidence for each crop, disease, formulation, "
            "dose, and waiting period. Confirm any pesticide choice with the current official label "
            "and a local agricultural officer.\n\n"
        )

        advisory_md += "### 3. Statutory Disclaimer & Citations\n"
        advisory_md += (
            "> [!IMPORTANT]\n"
            "> **Disclaimer**: These local knowledge files are retrieval references; their provenance and current validity have not been independently verified. They are not current product labels. "
            "Follow the applicable current official product label and local agricultural guidance. "
            "Always consult your local TNAU Krishi Vigyan Kendra (KVK) or Assistant Agricultural Officer prior to pesticide application.\n\n"
            f"**Citations**:\n"
        )
        for p in passages:
            advisory_md += f"- **[{p['source_file']}, Page {p['source_page']}]** `{p['section_title']}` (Similarity Score: {p['similarity_score']:.3f})\n"

        return {
            "status": "SUCCESS",
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "severity_band": severity_band,
            "retrieved_passages": passages,
            "citations": list(source_citations),
            "provenance_status": "local_reference_not_independently_verified",
            "advisory_markdown": advisory_md
        }

if __name__ == "__main__":
    rag = GroundedAdvisoryRAG()
    res = rag.get_advisory("Bacterial leaf blight", 0.92, "Severe", 45.0)
    print(res["advisory_markdown"])
