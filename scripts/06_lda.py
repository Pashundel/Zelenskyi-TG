import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.lda_model
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("lda_grid")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def read_csv_with_fallback(path: str, preferred_encoding: str) -> pd.DataFrame:
    encodings = [preferred_encoding, "utf-8-sig", "utf-8", "cp1251", "cp1252"]
    for enc in dict.fromkeys(encodings):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"Unable to decode {path}")


def parse_json_list(raw_value) -> List[str]:
    if isinstance(raw_value, list):
        return [str(x) for x in raw_value]
    if not isinstance(raw_value, str) or not raw_value.strip():
        return []
    try:
        parsed = json.loads(raw_value)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        return []
    return []


def build_topic_words(
    lda: LatentDirichletAllocation,
    feature_names: List[str],
    top_n: int
) -> pd.DataFrame:
    rows = []
    for topic_idx, weights in enumerate(lda.components_):
        top_ids = weights.argsort()[::-1][:top_n]
        for rank, term_id in enumerate(top_ids, start=1):
            rows.append(
                {
                    "topic_id": topic_idx,
                    "rank": rank,
                    "term": feature_names[term_id],
                    "weight": float(weights[term_id]),
                }
            )
    return pd.DataFrame(rows)


def extract_topics_for_coherence(
    lda: LatentDirichletAllocation,
    feature_names: List[str],
    top_n: int
) -> List[List[str]]:
    topics = []
    for weights in lda.components_:
        top_ids = weights.argsort()[::-1][:top_n]
        topics.append([feature_names[i] for i in top_ids])
    return topics


def compute_cv_coherence(
    topic_words: List[List[str]],
    token_lists: List[List[str]],
    dictionary: Dictionary
) -> float:
    cm = CoherenceModel(
        topics=topic_words,
        texts=token_lists,
        dictionary=dictionary,
        coherence="c_v",
    )
    return float(cm.get_coherence())


def save_coherence_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["k_topics"], metrics_df["coherence_cv"], marker="o")
    plt.xlabel("Number of topics (k)")
    plt.ylabel("C_v coherence")
    plt.title("LDA coherence by number of topics")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_perplexity_plot(metrics_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(metrics_df["k_topics"], metrics_df["perplexity"], marker="o")
    plt.xlabel("Number of topics (k)")
    plt.ylabel("Perplexity")
    plt.title("LDA perplexity by number of topics")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def pick_candidate_ks(metrics_df: pd.DataFrame, forced_k: int | None, n_candidates: int = 4) -> List[int]:
    candidates = set()

    # Top k by coherence
    top_coh = metrics_df.sort_values("coherence_cv", ascending=False).head(n_candidates)["k_topics"].tolist()
    candidates.update(top_coh)

    # Best by perplexity
    best_perp = metrics_df.sort_values("perplexity", ascending=True).head(1)["k_topics"].tolist()
    candidates.update(best_perp)

    # Forced k, if user wants to inspect a specific one like 15
    if forced_k is not None:
        candidates.add(forced_k)

    return sorted(candidates)


def save_topics_and_doc_topics(
    lda: LatentDirichletAllocation,
    best_k: int,
    feature_names: List[str],
    dtm,
    kept_idx: List[int],
    df: pd.DataFrame,
    outdir: Path,
    output_encoding: str,
    top_words: int
) -> Tuple[Path, Path]:
    topics_df = build_topic_words(lda, list(feature_names), top_words)
    topics_path = outdir / f"lda_topics_k{best_k}.csv"
    topics_df.to_csv(topics_path, index=False, encoding=output_encoding)

    best_doc_topic = lda.transform(dtm)
    doc_topic_cols = [f"topic_{i}" for i in range(best_k)]
    doc_topic_df = pd.DataFrame(best_doc_topic, columns=doc_topic_cols)
    doc_topic_df["source_row_id"] = kept_idx
    doc_topic_df["dominant_topic"] = doc_topic_df[doc_topic_cols].values.argmax(axis=1)
    doc_topic_df["dominant_topic_prob"] = doc_topic_df[doc_topic_cols].max(axis=1)

    if "id" in df.columns:
        doc_topic_df["post_id"] = [df.iloc[i]["id"] for i in kept_idx]

    doc_topics_path = outdir / f"lda_doc_topics_k{best_k}.csv"
    doc_topic_df.to_csv(doc_topics_path, index=False, encoding=output_encoding)

    return topics_path, doc_topics_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LDA grid search with perplexity + C_v coherence + pyLDAvis.")
    parser.add_argument("--input", default="posts_for_lda_unigrams")
    parser.add_argument("--input-encoding", default="utf-8-sig")
    parser.add_argument("--tokens-col", default="tokens_lda_json")
    parser.add_argument("--outdir", default="lda_outputs_cv")

    parser.add_argument("--min-topics", type=int, default=2)
    parser.add_argument("--max-topics", type=int, default=24)
    parser.add_argument("--topic-step", type=int, default=1)

    parser.add_argument("--min-doc-tokens", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--max-df", type=float, default=1.0)
    parser.add_argument("--min-df", type=int, default=1)

    parser.add_argument("--lda-max-iter", type=int, default=100)
    parser.add_argument("--learning-method", default="batch", choices=["batch", "online"])
    parser.add_argument("--random-state", type=int, default=42)

    parser.add_argument("--top-words", type=int, default=15)
    parser.add_argument("--coherence-topn", type=int, default=15)

    parser.add_argument("--inspect-k", type=int, default=15, help="Specific k to always include among pyLDAvis candidates.")
    parser.add_argument("--num-candidates", type=int, default=4)

    parser.add_argument("--output-encoding", default="utf-8-sig")
    parser.add_argument("--log", default="run_lda_grid_log.txt")
    args = parser.parse_args()

    logger = setup_logger(args.log)
    started = time.time()

    input_path = args.input
    if not Path(input_path).exists() and Path(f"{input_path}.csv").exists():
        input_path = f"{input_path}.csv"

    df = read_csv_with_fallback(input_path, args.input_encoding)
    if args.tokens_col not in df.columns:
        raise ValueError(f"Column '{args.tokens_col}' not found in {input_path}")

    token_lists_all = [parse_json_list(v) for v in df[args.tokens_col].fillna("")]
    token_counts = [len(toks) for toks in token_lists_all]
    keep_mask = [count >= args.min_doc_tokens for count in token_counts]
    kept_idx = [i for i, ok in enumerate(keep_mask) if ok]

    if not kept_idx:
        raise ValueError("No documents left after min token filter. Lower --min-doc-tokens.")

    token_lists = [token_lists_all[i] for i in kept_idx]
    docs_joined = [" ".join(toks) for toks in token_lists]

    logger.info("Input rows: %s", len(df))
    logger.info("Rows used for LDA (>= %s tokens): %s", args.min_doc_tokens, len(docs_joined))

    vectorizer = CountVectorizer(
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
        max_features=args.max_features,
        max_df=args.max_df,
        min_df=args.min_df,
    )
    dtm = vectorizer.fit_transform(docs_joined)
    feature_names = vectorizer.get_feature_names_out()
    logger.info("DTM shape: %s docs x %s terms", dtm.shape[0], dtm.shape[1])

    # Gensim dictionary for coherence
    dictionary = Dictionary(token_lists)
    logger.info("Gensim dictionary size before filtering: %s", len(dictionary))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    models_by_k = {}

    best_k_by_perplexity = None
    best_perplexity = None

    best_k_by_coherence = None
    best_coherence = None

    for k in range(args.min_topics, args.max_topics + 1, args.topic_step):
        logger.info("Training LDA with k=%s topics...", k)

        lda = LatentDirichletAllocation(
            n_components=k,
            max_iter=args.lda_max_iter,
            learning_method=args.learning_method,
            random_state=args.random_state,
            evaluate_every=-1,
            n_jobs=-1,
        )

        lda.fit(dtm)

        log_likelihood = float(lda.score(dtm))
        perplexity = float(lda.perplexity(dtm))

        topic_words = extract_topics_for_coherence(lda, list(feature_names), args.coherence_topn)
        coherence_cv = compute_cv_coherence(topic_words, token_lists, dictionary)

        metrics_rows.append(
            {
                "k_topics": k,
                "log_likelihood": log_likelihood,
                "perplexity": perplexity,
                "coherence_cv": coherence_cv,
                "n_docs": dtm.shape[0],
                "vocab_size": dtm.shape[1],
            }
        )

        models_by_k[k] = lda

        logger.info(
            "k=%s | coherence_cv=%.4f | perplexity=%.4f | log_likelihood=%.4f",
            k, coherence_cv, perplexity, log_likelihood
        )

        if best_perplexity is None or perplexity < best_perplexity:
            best_perplexity = perplexity
            best_k_by_perplexity = k

        if best_coherence is None or coherence_cv > best_coherence:
            best_coherence = coherence_cv
            best_k_by_coherence = k

    metrics_df = pd.DataFrame(metrics_rows).sort_values("k_topics")
    metrics_path = outdir / "lda_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding=args.output_encoding)

    if best_k_by_coherence is None or best_k_by_perplexity is None:
        raise RuntimeError("No LDA models were produced.")

    logger.info("Best k by coherence: %s (%.4f)", best_k_by_coherence, best_coherence)
    logger.info("Best k by perplexity: %s (%.4f)", best_k_by_perplexity, best_perplexity)

    # Save plots
    coherence_plot_path = outdir / "lda_coherence_plot.png"
    save_coherence_plot(metrics_df, coherence_plot_path)

    perplexity_plot_path = outdir / "lda_perplexity_plot.png"
    save_perplexity_plot(metrics_df, perplexity_plot_path)

    # Save outputs for best-by-coherence model
    best_model = models_by_k[best_k_by_coherence]
    topics_path, doc_topics_path = save_topics_and_doc_topics(
        lda=best_model,
        best_k=best_k_by_coherence,
        feature_names=list(feature_names),
        dtm=dtm,
        kept_idx=kept_idx,
        df=df,
        outdir=outdir,
        output_encoding=args.output_encoding,
        top_words=args.top_words,
    )

    # Save summary
    summary = {
        "input": str(input_path),
        "tokens_col": args.tokens_col,
        "rows_total": len(df),
        "rows_used": len(kept_idx),
        "min_doc_tokens": args.min_doc_tokens,
        "vocab_size": int(dtm.shape[1]),
        "top_words_per_topic": args.top_words,
        "best_k_by_coherence": best_k_by_coherence,
        "best_coherence_cv": best_coherence,
        "best_k_by_perplexity": best_k_by_perplexity,
        "best_perplexity": best_perplexity,
        "inspect_k": args.inspect_k,
    }
    summary_path = outdir / "lda_model_selection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Pick candidate k values for pyLDAvis
    candidate_ks = pick_candidate_ks(
        metrics_df=metrics_df,
        forced_k=args.inspect_k,
        n_candidates=args.num_candidates
    )
    logger.info("Candidate k values for pyLDAvis: %s", candidate_ks)

    # Save pyLDAvis HTMLs
    for k in candidate_ks:
        lda_model = models_by_k[k]
        logger.info("Building pyLDAvis for k=%s ...", k)
        vis = pyLDAvis.lda_model.prepare(
            lda_model,
            dtm,
            vectorizer,
            mds="pcoa"
        )
        html_path = outdir / f"pyldavis_k{k}.html"
        pyLDAvis.save_html(vis, str(html_path))

    logger.info("Saved metrics: %s", metrics_path)
    logger.info("Saved coherence plot: %s", coherence_plot_path)
    logger.info("Saved perplexity plot: %s", perplexity_plot_path)
    logger.info("Saved best topics (by coherence): %s", topics_path)
    logger.info("Saved doc-topic matrix (by coherence): %s", doc_topics_path)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Done in %.2f minutes", (time.time() - started) / 60)


if __name__ == "__main__":
    main()