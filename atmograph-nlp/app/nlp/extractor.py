"""spaCy NER plus configurable Matcher rules for supply-chain entities.

The extractor combines two evidence sources:

* spaCy's statistical NER (``ORG``, ``GPE``, ``LOC``, ``FAC``, ``PRODUCT`` ...)
* spaCy ``PhraseMatcher``/``Matcher`` rules for ports, facilities, materials,
  industries and products that the small English model does not recognise.

Overlaps are resolved deterministically with a length-plus-rule-priority score,
so a long organisation name is not fragmented by a short keyword rule while a
domain phrase such as ``Kaohsiung Terminal`` can override a generic NER label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc

from app.config import Settings, get_settings
from app.nlp.model_loader import get_nlp, parse_text
from app.nlp.normalizer import canonical_key, normalize_entity_name
from app.nlp.patterns import Pattern, build_patterns
from app.nlp.taxonomy import (
    COMPANY_ACTION_LEMMAS,
    COMPANY_CUE_TOKENS,
    COUNTRY_ALIASES,
    COUNTRY_NAMES,
    HEADLINE_STOPWORDS,
    KNOWN_LOCATION_NAMES,
    ORGANIZATION_KEYWORDS,
    RULE_LABELS,
    SPACY_LABEL_TYPES,
    WEATHER_PREFIXES,
    EntityType,
)
from app.schemas.article import Article
from app.schemas.nlp import EntityAnnotation, ExtractionMethod, SourceField
from app.utils.logger import get_logger

logger = get_logger(__name__)

_RULE_SCORE_BONUS = 15
_WEATHER_ENTITY_TYPES: frozenset[str] = frozenset(
    {"PRODUCT", "PERSON", "ORGANIZATION", "COMPANY"}
)
_WORD_EDGE_PUNCTUATION = " \t\n\r.,;:!?()[]{}<>\"'“”‘’«»-"

_COUNTRY_KEYS: frozenset[str] = frozenset(
    {canonical_key(name) for name in COUNTRY_NAMES}
    | {canonical_key(alias) for alias in COUNTRY_ALIASES}
)
_LOCATION_KEYS: frozenset[str] = frozenset(
    canonical_key(name) for name in KNOWN_LOCATION_NAMES
)
_ORG_KEYWORD_TOKENS: frozenset[str] = frozenset(
    keyword for keyword in ORGANIZATION_KEYWORDS if " " not in keyword
)
_ORG_KEYWORD_PHRASES: tuple[str, ...] = tuple(
    keyword for keyword in ORGANIZATION_KEYWORDS if " " in keyword
)


@dataclass(frozen=True, slots=True)
class RuleSpan:
    """A character-offset span produced by a spaCy Matcher."""

    start_char: int
    end_char: int
    label: str
    text: str


@dataclass(frozen=True, slots=True)
class _Candidate:
    start: int
    end: int
    entity_type: EntityType
    method: ExtractionMethod
    raw_label: str
    text: str
    score: int


class RuleMatcher:
    """Compile phrase and token patterns once and reuse them for every article."""

    def __init__(self, nlp: Language, patterns: Sequence[Pattern]) -> None:
        self._nlp = nlp
        self._phrase_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._token_matcher = Matcher(nlp.vocab)
        self._labels: dict[int, str] = {}

        phrase_groups: dict[str, list[Doc]] = {}
        token_groups: dict[str, list[list[dict]]] = {}
        for item in patterns:
            label = str(item["label"])
            raw_pattern = item["pattern"]
            if isinstance(raw_pattern, str):
                phrase_groups.setdefault(label, []).append(nlp.make_doc(raw_pattern))
            else:
                token_groups.setdefault(label, []).append(raw_pattern)

        for label, docs in phrase_groups.items():
            self._phrase_matcher.add(label, docs)
            self._labels[nlp.vocab.strings[label]] = label
        for label, token_patterns in token_groups.items():
            self._token_matcher.add(label, token_patterns)
            self._labels[nlp.vocab.strings[label]] = label

        self.pattern_count = len(patterns)

    def find(self, doc: Doc) -> list[RuleSpan]:
        spans: list[RuleSpan] = []
        for matcher in (self._phrase_matcher, self._token_matcher):
            for match_id, start, end in matcher(doc):
                label = self._labels.get(match_id)
                if label is None:
                    continue
                start_char = doc[start].idx
                end_char = doc[end - 1].idx + len(doc[end - 1].text)
                spans.append(
                    RuleSpan(
                        start_char=start_char,
                        end_char=end_char,
                        label=label,
                        text=doc.text[start_char:end_char],
                    )
                )
        return spans


class ArticleEntityExtractor:
    """Extract normalized, offset-preserving entities from one article."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        nlp: Language | None = None,
        patterns: Sequence[Pattern] | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_name = self._settings.spacy_model
        self._nlp = nlp or get_nlp(self._model_name)
        compiled_patterns = (
            list(patterns)
            if patterns is not None
            else build_patterns(self._settings.nlp_patterns_path)
        )
        self._rules = RuleMatcher(self._nlp, compiled_patterns)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def pattern_count(self) -> int:
        return self._rules.pattern_count

    @property
    def pipeline_components(self) -> list[str]:
        return list(self._nlp.pipe_names)

    def extract(self, article: Article) -> list[EntityAnnotation]:
        """Return entities from the title followed by the description."""
        entities: list[EntityAnnotation] = []
        for field, text in (
            ("title", article.title),
            ("description", article.description),
        ):
            if text:
                entities.extend(
                    self.extract_field(
                        text,
                        article_id=article.article_id,
                        source_field=field,
                    )
                )
        return entities

    def extract_field(
        self,
        text: str,
        *,
        article_id: str,
        source_field: SourceField,
    ) -> list[EntityAnnotation]:
        """Extract entities from one article field with exact offsets."""
        if not text:
            return []

        doc = parse_text(self._nlp, text)
        candidates = [
            candidate
            for candidate in (*self._spacy_candidates(doc), *self._rule_candidates(doc))
            if self._is_plausible(candidate)
        ]
        selected = self._select(candidates)

        annotations: list[EntityAnnotation] = []
        seen: set[tuple[str, str, int, int]] = set()
        for candidate in selected:
            trimmed = _trim_offsets(text, candidate.start, candidate.end)
            if trimmed is None:
                continue
            start, end = trimmed
            surface = text[start:end]
            if len(surface) < 2 or not any(character.isalnum() for character in surface):
                continue
            if (
                candidate.entity_type in _WEATHER_ENTITY_TYPES
                and surface.split()[0].lower() in WEATHER_PREFIXES
            ):
                continue

            normalized = normalize_entity_name(surface, candidate.entity_type)
            if not normalized:
                continue
            signature = (
                candidate.entity_type,
                canonical_key(normalized),
                start,
                end,
            )
            if signature in seen:
                continue
            seen.add(signature)

            annotations.append(
                EntityAnnotation(
                    article_id=article_id,
                    text=surface,
                    normalized_name=normalized,
                    entity_type=candidate.entity_type,
                    source_field=source_field,
                    start_char=start,
                    end_char=end,
                    extraction_method=candidate.method,
                    raw_label=candidate.raw_label,
                )
            )

        annotations.sort(key=lambda entity: (entity.start_char, entity.end_char))
        return annotations

    # -- candidate generation ---------------------------------------------
    def _spacy_candidates(self, doc: Doc) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for entity in doc.ents:
            entity_type = self._map_spacy_entity(entity, doc)
            if entity_type is None:
                continue
            candidates.append(
                _Candidate(
                    start=entity.start_char,
                    end=entity.end_char,
                    entity_type=entity_type,
                    method="spacy",
                    raw_label=entity.label_,
                    text=entity.text,
                    score=len(entity.text),
                )
            )
        return candidates

    def _rule_candidates(self, doc: Doc) -> list[_Candidate]:
        candidates: list[_Candidate] = []
        for span in self._rules.find(doc):
            entity_type = RULE_LABELS.get(span.label)
            if entity_type is None:
                continue
            candidates.append(
                _Candidate(
                    start=span.start_char,
                    end=span.end_char,
                    entity_type=entity_type,
                    method="rule",
                    raw_label=span.label,
                    text=span.text,
                    score=len(span.text) + _RULE_SCORE_BONUS,
                )
            )
        return candidates

    def _map_spacy_entity(self, entity, doc: Doc) -> EntityType | None:
        label = entity.label_
        if label == "ORG":
            return self._classify_organization(entity.text)
        if label == "GPE":
            return self._classify_gpe(entity, doc)
        return SPACY_LABEL_TYPES.get(label)

    @staticmethod
    def _classify_organization(text: str) -> EntityType:
        return "ORGANIZATION" if ArticleEntityExtractor._has_org_keyword(text) else "COMPANY"

    @staticmethod
    def _has_org_keyword(text: str) -> bool:
        key = canonical_key(text)
        tokens = set(key.split())
        if tokens & _ORG_KEYWORD_TOKENS:
            return True
        return any(phrase in key for phrase in _ORG_KEYWORD_PHRASES)

    @staticmethod
    def _is_plausible(candidate: _Candidate) -> bool:
        """Reject headline fragments that spaCy occasionally labels as names."""
        words = {
            word.strip(_WORD_EDGE_PUNCTUATION).lower()
            for word in candidate.text.split()
            if word.strip(_WORD_EDGE_PUNCTUATION)
        }
        if not words:
            return False

        if candidate.entity_type in _WEATHER_ENTITY_TYPES and (
            candidate.text.split()[0].lower() in WEATHER_PREFIXES
        ):
            return False

        if candidate.entity_type == "PERSON":
            if words & HEADLINE_STOPWORDS:
                return False
            return len(words) <= 4

        if candidate.entity_type in {"ORGANIZATION", "COMPANY"}:
            has_org_keyword = ArticleEntityExtractor._has_org_keyword(candidate.text)
            if words & HEADLINE_STOPWORDS and not has_org_keyword:
                return False
            if len(words) > 4 and not has_org_keyword:
                return False

        return True

    @staticmethod
    def _classify_gpe(entity, doc: Doc) -> EntityType:
        key = canonical_key(entity.text)
        if key in _COUNTRY_KEYS:
            return "COUNTRY"
        if key in _LOCATION_KEYS:
            return "LOCATION"

        previous = doc[entity.start - 1] if entity.start > 0 else None
        following = doc[entity.end] if entity.end < len(doc) else None
        if previous is not None and previous.lower_ in COMPANY_CUE_TOKENS:
            return "COMPANY"
        if following is not None and following.lemma_.lower() in COMPANY_ACTION_LEMMAS:
            return "COMPANY"
        return "LOCATION"

    # -- overlap resolution ------------------------------------------------
    @staticmethod
    def _select(candidates: Sequence[_Candidate]) -> list[_Candidate]:
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.score,
                0 if candidate.method == "rule" else 1,
                candidate.start,
                -(candidate.end - candidate.start),
            ),
        )
        selected: list[_Candidate] = []
        for candidate in ordered:
            if any(_overlaps(candidate, chosen) for chosen in selected):
                continue
            selected.append(candidate)
        selected.sort(key=lambda candidate: (candidate.start, candidate.end))
        return selected


def _overlaps(left: _Candidate, right: _Candidate) -> bool:
    return left.start < right.end and right.start < left.end


def _trim_offsets(text: str, start: int, end: int) -> tuple[int, int] | None:
    raw = text[start:end]
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped == raw:
        return start, end
    leading = raw.find(stripped)
    return start + leading, start + leading + len(stripped)


__all__ = ["ArticleEntityExtractor", "RuleMatcher", "RuleSpan"]
