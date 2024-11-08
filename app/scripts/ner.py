from skillNer.skill_extractor_class import SkillExtractor
from typing import List

def extract_skills_from_text(model: SkillExtractor, text: str) -> List[str]:
    """
    Performs Named Entity Recognition (NER) to extract skill-related 
    words from an input text.
    """
    annotations = model.annotate(text)
    results = annotations.get("results")
    matches = [match["doc_node_value"] for match in results["full_matches"]] if results else []
    ngrams = [ngram["doc_node_value"] for ngram in results["ngram_scored"]] if results else []

    return matches + ngrams
