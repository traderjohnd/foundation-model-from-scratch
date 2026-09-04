"""Canonical WikiText-103 loading, normalization, and article reconstruction.

The core normalization and reconstruction logic in this module is extracted from the
verified Notebook 01 data-preparation/corpus-audit implementation. Keep behavioral
changes separate from extraction-only changes so the audited document counts remain a
clear regression contract.
"""

import re

from datasets import load_dataset


DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"

AUDIT_SPLITS = ("train", "validation")
EXPECTED_ROW_COUNTS = {
    "train": 1_801_350,
    "validation": 3_760,
    "test": 4_358,
}
EXPECTED_ARTICLE_COUNTS = {
    "train": 28_472,
    "validation": 60,
}


def load_pinned_wikitext():
    """Load the immutable WikiText-103 revision used by the project."""
    dataset = load_dataset(
        DATASET_ID, DATASET_CONFIG, revision=DATASET_REVISION
    )
    for split_name, expected_rows in EXPECTED_ROW_COUNTS.items():
        actual_rows = dataset[split_name].num_rows
        assert actual_rows == expected_rows, (
            f"Unexpected {split_name} row count: {actual_rows:,} != {expected_rows:,}"
        )
    return dataset


NORMALIZATION_RULES = (
    (re.compile(r"\s*@-@\s*"), "-"),
    (re.compile(r"\s*@,@\s*"), ","),
    (re.compile(r"\s*@\.@\s*"), "."),
    (re.compile(r"\b(\w+)\s+'\s*(t|s|re|ve|ll|d|m)\b", re.IGNORECASE), r"\1'\2"),
    (re.compile(r"([$£€])\s+(?=\d)"), r"\1"),
    (re.compile(r"\b(\d{1,2})\s*:\s*(\d{2})\b"), r"\1:\2"),
    (re.compile(r"(\d)\s+–\s+(\d)"), r"\1–\2"),
    (re.compile(r"\s+([,.;:!?%])"), r"\1"),
    (re.compile(r"([(\[\{])\s+"), r"\1"),
    (re.compile(r"\s+([)\]\}])"), r"\1"),
    (re.compile(r"[ \t]+"), " "),
)


def tighten_spaced_double_quote_pairs(text):
    quote = '"'
    escaped_quote = re.escape(quote)
    paired_pattern = re.compile(
        rf"{escaped_quote}\s+(.+?)\s+{escaped_quote}"
    )
    return paired_pattern.sub(
        lambda match: f"{quote}{match.group(1).strip()}{quote}", text
    )


HEADING_FRAME_PATTERN = re.compile(
    r"^\s*(?P<left>=+(?:\s+=+)*)\s+"
    r"(?P<title>.*?)"
    r"\s+(?P<right>=+(?:\s+=+)*)\s*$"
)


def parse_heading_row(text):
    match = HEADING_FRAME_PATTERN.fullmatch(text)
    if match is None:
        return None
    left_level = match.group("left").count("=")
    right_level = match.group("right").count("=")
    title = match.group("title").strip()
    if left_level != right_level or not title:
        return None
    return {"level": left_level, "title": title}


def normalize_wikitext_fragment(text):
    normalized = text
    for pattern, replacement in NORMALIZATION_RULES:
        normalized = pattern.sub(replacement, normalized)
    normalized = tighten_spaced_double_quote_pairs(normalized)
    return normalized.strip()


def normalize_wikitext_text(text):
    if not isinstance(text, str):
        raise TypeError(f"Expected str, received {type(text).__name__}")
    heading = parse_heading_row(text)
    if heading is None:
        return normalize_wikitext_fragment(text)
    marker = " ".join("=" for _ in range(heading["level"]))
    title = normalize_wikitext_fragment(heading["title"])
    return f"{marker} {title} {marker}"


NORMALIZATION_TEST_CASES = {
    "well @-@ known": "well-known",
    "3 @.@ 5 million": "3.5 million",
    "1 @,@ 000 people": "1,000 people",
    "( 1987 )": "(1987)",
    "don 't": "don't",
    "café 's tables": "café's tables",
    "the players ' hopes": "the players ' hopes",
    "$ 3 @.@ 5 million": "$3.5 million",
    "The meeting ran from 12 : 30 to 13 : 05 .": (
        "The meeting ran from 12:30 to 13:05."
    ),
    'the " Nameless ", a penal unit': 'the "Nameless", a penal unit',
    "the ' Nameless ' unit": "the ' Nameless ' unit",
    "the players ' hopes and coaches ' plans": (
        "the players ' hopes and coaches ' plans"
    ),
    "U.S.": "U.S.",
    "2011 – 12 season": "2011–12 season",
    "= ... Thirteen Years Later =": "= ... Thirteen Years Later =",
    "= ? ( film ) =": "= ? (film) =",
    "= = Internal Section = =": "= = Internal Section = =",
}


def run_normalization_self_test():
    """Run the same 17 normalization regression cases verified in Notebook 01."""
    results = []
    for raw_text, expected_text in NORMALIZATION_TEST_CASES.items():
        actual_text = normalize_wikitext_text(raw_text)
        result = {
            "input": raw_text,
            "expected": expected_text,
            "actual": actual_text,
            "passed": actual_text == expected_text,
        }
        results.append(result)
    failed = [result for result in results if not result["passed"]]
    assert not failed, failed
    return results


def normalize_text_batch(batch):
    return {
        "text": [normalize_wikitext_text(text) for text in batch["text"]]
    }


def normalize_development_splits(raw_dataset, batch_size=1_000):
    """Normalize only train/validation; intentionally leave test untouched."""
    return {
        split_name: raw_dataset[split_name].map(
            normalize_text_batch,
            batched=True,
            batch_size=batch_size,
            desc=f"Normalize {split_name}",
        )
        for split_name in AUDIT_SPLITS
    }


def iter_text_rows(split, batch_size=10_000):
    for batch in split.iter(batch_size=batch_size):
        yield from batch["text"]


def iter_with_neighbors(rows):
    iterator = iter(rows)
    previous = None
    current = next(iterator, None)
    following = next(iterator, None)
    row_index = 0
    while current is not None:
        yield row_index, previous, current, following
        previous = current
        current = following
        following = next(iterator, None)
        row_index += 1


def is_blank_row(text):
    return text is not None and not text.strip()


def is_raw_article_start(previous_raw, raw_text, following_raw):
    heading = parse_heading_row(raw_text)
    return (
        heading is not None
        and heading["level"] == 1
        and is_blank_row(previous_raw)
        and is_blank_row(following_raw)
    )


def reconstruct_articles(raw_text_rows, normalized_text_rows, split_name):
    articles = []
    current = None
    last_row_index = -1
    paired_rows = zip(raw_text_rows, normalized_text_rows, strict=True)
    for row_index, previous, row_pair, following in iter_with_neighbors(paired_rows):
        last_row_index = row_index
        raw_text, normalized_text = row_pair
        previous_raw = previous[0] if previous is not None else None
        following_raw = following[0] if following is not None else None
        if is_raw_article_start(previous_raw, raw_text, following_raw):
            if current is not None:
                current["end_row"] = row_index - 1
                current["text"] = "\n".join(current.pop("rows"))
                articles.append(current)

            raw_heading = parse_heading_row(raw_text)
            article_index = len(articles)
            current = {
                "article_id": (
                    f"{split_name}:article-{article_index:05d}:row-{row_index}"
                ),
                "title": normalize_wikitext_fragment(raw_heading["title"]),
                "start_row": row_index,
                "rows": [normalized_text],
            }
        elif current is None:
            if raw_text.strip():
                raise ValueError(
                    f"Nonblank row {row_index} appears before the first article heading."
                )
        else:
            current["rows"].append(normalized_text)

    if current is not None:
        current["end_row"] = last_row_index
        current["text"] = "\n".join(current.pop("rows"))
        articles.append(current)

    return articles


def reconstruct_development_articles(raw_dataset, normalized_development):
    """Rebuild train/validation article records and enforce the audited counts."""
    articles_by_split = {
        split_name: reconstruct_articles(
            iter_text_rows(raw_dataset[split_name]),
            iter_text_rows(normalized_development[split_name]),
            split_name,
        )
        for split_name in AUDIT_SPLITS
    }
    for split_name, expected_count in EXPECTED_ARTICLE_COUNTS.items():
        actual_count = len(articles_by_split[split_name])
        assert actual_count == expected_count, (
            f"Unexpected {split_name} article count: "
            f"{actual_count:,} != {expected_count:,}"
        )
    return articles_by_split
