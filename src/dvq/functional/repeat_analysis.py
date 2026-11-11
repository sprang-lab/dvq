import pandas as pd
import logging
import re
from tqdm import tqdm
from typing import List, Optional

try:
    from pytrf import GTRFinder
except ImportError:
    GTRFinder = None


def _fallback_find_tandem_repeats(seq: str, min_motif: int = 1, max_motif: int = 30, min_repeat: int = 3, min_total_length: int = 6):
    """Pure-Python exact tandem repeat scanner used when pytrf is not installed."""
    results = []
    n = len(seq)
    for m_len in range(min_motif, min(max_motif, n // min_repeat) + 1):
        i = 0
        while i <= n - m_len * min_repeat:
            motif = seq[i : i + m_len]
            if len(set(motif)) == 1 and m_len > 1:
                i += 1
                continue
            r_num = 1
            while i + (r_num + 1) * m_len <= n and seq[i + r_num * m_len : i + (r_num + 1) * m_len] == motif:
                r_num += 1
            r_len = r_num * m_len
            if r_num >= min_repeat and r_len >= min_total_length:
                results.append((None, i + 1, i + r_len, motif, m_len, r_num, r_len))
                i += r_len
            else:
                i += 1
    return results


def find_genomic_repeats(
    sequences: List[str],
    ids: Optional[List[str]] = None,
    min_motif: int = 1,
    max_motif: int = 30,
    min_repeat: int = 3,
    min_total_length: int = 6,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Analyses a list of sequences for tandem repeats and returns a detailed DataFrame.
    """
    if ids is None:
        ids = [f"seq_{i}" for i in range(len(sequences))]

    if len(sequences) != len(ids):
        raise ValueError("The length of sequences and ids must match.")

    all_repeats = []
    iterator = zip(ids, sequences)

    if show_progress:
        iterator = tqdm(iterator, total=len(sequences), desc="Finding Repeats")

    for seq_id, seq in iterator:
        if not seq or pd.isna(seq):
            continue

        try:
            if GTRFinder is not None:
                finder = GTRFinder(
                    seq_id,
                    str(seq),
                    min_motif=min_motif,
                    max_motif=max_motif,
                    min_repeat=min_repeat,
                    min_length=min_total_length
                )
                repeat_rows = finder.as_list()
            else:
                repeat_rows = _fallback_find_tandem_repeats(
                    str(seq),
                    min_motif=min_motif,
                    max_motif=max_motif,
                    min_repeat=min_repeat,
                    min_total_length=min_total_length,
                )

            for _, start, end, motif, m_len, r_num, r_len in repeat_rows:
                all_repeats.append({
                    'Sequence_ID': seq_id,
                    'Start': start,
                    'End': end,
                    'Motif': motif,
                    'Motif Length': m_len,
                    'Repeat Number': r_num,
                    'Repeat Length': r_len,
                })
        except Exception as e:
            logging.error(f"Error processing sequence {seq_id}: {e}")

    df_results = pd.DataFrame(all_repeats)

    if not df_results.empty:
        print("\n--- Top 10 Common Repeat-Motif Pairs ---")
        stats = df_results.groupby(['Repeat Number', 'Motif']).size().reset_index(name='Count')
        top_10 = stats.sort_values(by='Count', ascending=False).head(10)
        print(top_10.to_string(index=False))
        print("-" * 40)
    else:
        logging.warning("No repeats found in the provided sequences.")

    return df_results
