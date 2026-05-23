"""Multi-Agent Teacher: five specialized agents."""
from .structure_parser import StructureParserAgent
from .key_element_extractor import KeyElementExtractorAgent
from .evidence_retriever import EvidenceRetrieverAgent
from .verification_reasoner import VerificationReasonerAgent
from .review_synthesizer import ReviewSynthesizerAgent

__all__ = [
    "StructureParserAgent",
    "KeyElementExtractorAgent",
    "EvidenceRetrieverAgent",
    "VerificationReasonerAgent",
    "ReviewSynthesizerAgent",
]
