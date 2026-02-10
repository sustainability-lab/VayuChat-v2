"""
Test harness for VayuChat function-calling vs code-gen benchmark.

Usage:
    python test_harness.py                  # FC-only: test tool selection + timing
    python test_harness.py --full           # Also run code-gen for speed comparison
    python test_harness.py --dry-run        # Just print the question set, no API calls
    python test_harness.py --model gemini-2.0-flash   # Use a specific model
    python test_harness.py --filter plot_   # Only run questions expecting plot_* tools
    python test_harness.py --limit 10       # Run first N questions only

Output:  test_results.csv  +  console summary
"""

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from google import genai

import plotly.graph_objects as go
from analysis_tools import analysis_tools, FC_SYSTEM_PROMPT, dispatch

load_dotenv(override=True)

# ── Models ──────────────────────────────────────────────────────────────────
MODELS = {
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
}

# ── Questions ───────────────────────────────────────────────────────────────
# (question, expected_function, expected_output_type, category)
# expected_function can be a string or a list of acceptable function names
# expected_output_type: "text", "table", "plot", "fallback"

QUESTIONS = [
    # ────────────── get_statistics  (~20) ──────────────
    ("Which city has the highest average PM2.5?",
     "get_statistics", "text", "get_statistics"),
    ("Which state has the lowest average PM10?",
     "get_statistics", "text", "get_statistics"),
    ("What is the average PM2.5 in Delhi?",
     "get_statistics", "text", "get_statistics"),
    ("Which month has the highest pollution in Mumbai?",
     "get_statistics", "text", "get_statistics"),
    ("Top 5 cities by NO2 levels",
     "get_statistics", "text", "get_statistics"),
    ("What is the median PM2.5 across all cities?",
     "get_statistics", "text", "get_statistics"),
    ("Which city has the highest maximum PM2.5 in 2023?",
     "get_statistics", "text", "get_statistics"),
    ("Average ozone levels by state",
     ["get_statistics", "df_ranking"], "text", "get_statistics"),
    ("Which season has the highest PM2.5 in Delhi?",
     "get_statistics", "text", "get_statistics"),
    ("What is the average temperature in Mumbai?",
     "get_statistics", "text", "get_statistics"),
    ("Highest PM10 city in 2022",
     "get_statistics", "text", "get_statistics"),
    ("What is the standard deviation of PM2.5 across cities?",
     ["get_statistics", "df_ranking"], "text", "get_statistics"),
    ("Minimum PM2.5 by year",
     ["get_statistics", "df_ranking"], "text", "get_statistics"),
    ("Which station has the highest NO2?",
     "get_statistics", "text", "get_statistics"),
    ("Average SO2 levels in Tamil Nadu",
     "get_statistics", "text", "get_statistics"),
    ("Which city has the worst air quality?",
     "get_statistics", "text", "get_statistics"),
    ("What is the average wind speed in Delhi?",
     "get_statistics", "text", "get_statistics"),
    ("Top 3 most polluted states",
     "get_statistics", "text", "get_statistics"),
    ("Which year had the highest pollution?",
     "get_statistics", "text", "get_statistics"),
    ("Average humidity by season",
     ["get_statistics", "df_ranking"], "text", "get_statistics"),

    # ────────────── get_exceedances  (~10) ──────────────
    ("Which cities exceed the WHO PM2.5 guideline?",
     "get_exceedances", "text", "get_exceedances"),
    ("Cities above India PM2.5 standard of 60",
     "get_exceedances", "text", "get_exceedances"),
    ("States where PM10 exceeds 100",
     "get_exceedances", "text", "get_exceedances"),
    ("Cities below WHO PM2.5 guideline of 15",
     "get_exceedances", "text", "get_exceedances"),
    ("Which cities have PM2.5 above 80?",
     "get_exceedances", "text", "get_exceedances"),
    ("States exceeding WHO PM10 guideline in 2023",
     "get_exceedances", "text", "get_exceedances"),
    ("Cities with average NO2 above 40",
     "get_exceedances", "text", "get_exceedances"),
    ("Which areas have PM2.5 below 30?",
     "get_exceedances", "text", "get_exceedances"),
    ("Cities with ozone above 100",
     "get_exceedances", "text", "get_exceedances"),
    ("Which cities have SO2 levels exceeding 20?",
     "get_exceedances", "text", "get_exceedances"),

    # ────────────── get_ncap_analysis  (~5) ──────────────
    ("Which NCAP cities exceed air quality guidelines?",
     "get_ncap_analysis", "text", "get_ncap_analysis"),
    ("What are the top funded NCAP cities?",
     "get_ncap_analysis", "text", "get_ncap_analysis"),
    ("NCAP funding summary",
     "get_ncap_analysis", "text", "get_ncap_analysis"),
    ("Do NCAP cities meet India PM2.5 standards?",
     "get_ncap_analysis", "text", "get_ncap_analysis"),
    ("Which NCAP cities have PM10 above India guideline?",
     "get_ncap_analysis", "text", "get_ncap_analysis"),
    ("How much NCAP funding did Delhi receive vs Mumbai?",
     ["get_ncap_analysis", "plot_ncap"], "text", "get_ncap_analysis"),
    ("Which NCAP cities achieved the best PM2.5 reduction?",
     "get_ncap_analysis", "text", "get_ncap_analysis"),

    # ────────────── df_ranking  (~10) ──────────────
    ("List top 10 cities by PM2.5 levels",
     "df_ranking", "table", "df_ranking"),
    ("Rank all states by PM10",
     "df_ranking", "table", "df_ranking"),
    ("Show a table of cities ranked by NO2",
     "df_ranking", "table", "df_ranking"),
    ("Top 20 most polluted cities in a table",
     "df_ranking", "table", "df_ranking"),
    ("Rank months by PM2.5 levels in Delhi",
     "df_ranking", "table", "df_ranking"),
    ("List all cities by average ozone in a table",
     "df_ranking", "table", "df_ranking"),
    ("Table of states ranked by SO2",
     "df_ranking", "table", "df_ranking"),
    ("Top 5 stations by PM2.5 in 2023 table",
     "df_ranking", "table", "df_ranking"),
    ("Rank seasons by pollution levels",
     "df_ranking", "table", "df_ranking"),
    ("Show ranking table of cities by temperature",
     "df_ranking", "table", "df_ranking"),

    # ────────────── df_exceedances  (~5) ──────────────
    ("Show table of cities exceeding WHO PM2.5 guideline",
     "df_exceedances", "table", "df_exceedances"),
    ("List all cities with PM10 above 100 in a table",
     "df_exceedances", "table", "df_exceedances"),
    ("Table of states with PM2.5 above 60",
     "df_exceedances", "table", "df_exceedances"),
    ("Show cities where SO2 exceeds 40 as a table",
     "df_exceedances", "table", "df_exceedances"),
    ("List areas below WHO PM2.5 guideline in a table",
     "df_exceedances", "table", "df_exceedances"),

    # ────────────── plot_trend  (~15) ──────────────
    ("Plot monthly PM2.5 trends for Delhi",
     "plot_trend", "plot", "plot_trend"),
    ("Show yearly PM10 trend",
     "plot_trend", "plot", "plot_trend"),
    ("Graph PM2.5 trends for Mumbai and Delhi",
     "plot_trend", "plot", "plot_trend"),
    ("Plot monthly NO2 trend for 2023",
     "plot_trend", "plot", "plot_trend"),
    ("Chart yearly ozone trend for Karnataka",
     "plot_trend", "plot", "plot_trend"),
    ("Show PM2.5 trend from 2020 to 2024",
     "plot_trend", "plot", "plot_trend"),
    ("Visualize monthly PM10 trends in Maharashtra",
     "plot_trend", "plot", "plot_trend"),
    ("Plot CO trends over the years",
     "plot_trend", "plot", "plot_trend"),
    ("Graph NH3 monthly trend for Lucknow",
     "plot_trend", "plot", "plot_trend"),
    ("Plot PM2.5 yearly trend for all cities",
     "plot_trend", "plot", "plot_trend"),
    ("Show temperature trend in Delhi 2020 to 2023",
     "plot_trend", "plot", "plot_trend"),
    ("Chart SO2 trend in Gujarat",
     "plot_trend", "plot", "plot_trend"),
    ("Plot PM2.5 trends comparing Delhi and Mumbai",
     "plot_trend", "plot", "plot_trend"),
    ("Show yearly rainfall trend chart",
     "plot_trend", "plot", "plot_trend"),
    ("Visualize monthly wind speed trend",
     "plot_trend", "plot", "plot_trend"),

    # ────────────── plot_comparison  (~10) ──────────────
    ("Compare PM2.5 across seasons in a chart",
     "plot_comparison", "plot", "plot_comparison"),
    ("Plot seasonal comparison of PM10",
     "plot_comparison", "plot", "plot_comparison"),
    ("Compare weekday vs weekend pollution chart",
     "plot_comparison", "plot", "plot_comparison"),
    ("Chart comparing PM2.5 between Delhi, Mumbai, and Kolkata",
     "plot_comparison", "plot", "plot_comparison"),
    ("Plot state comparison of PM10 for Maharashtra and Gujarat",
     "plot_comparison", "plot", "plot_comparison"),
    ("Compare north vs south India PM2.5 in a plot",
     "plot_comparison", "plot", "plot_comparison"),
    ("Seasonal box plot of ozone",
     "plot_comparison", "plot", "plot_comparison"),
    ("Compare PM2.5 between winter and monsoon in a chart",
     "plot_comparison", "plot", "plot_comparison"),
    ("Show weekday vs weekend NO2 comparison chart",
     "plot_comparison", "plot", "plot_comparison"),
    ("Plot comparing PM2.5 across seasons in 2023",
     "plot_comparison", "plot", "plot_comparison"),

    # ────────────── plot_correlation  (~10) ──────────────
    ("Plot PM2.5 vs PM10 correlation",
     "plot_correlation", "plot", "plot_correlation"),
    ("Scatter plot of temperature vs PM2.5",
     "plot_correlation", "plot", "plot_correlation"),
    ("Show correlation between wind speed and PM2.5 chart",
     "plot_correlation", "plot", "plot_correlation"),
    ("Chart PM2.5 vs humidity",
     "plot_correlation", "plot", "plot_correlation"),
    ("Plot NO2 vs PM2.5 correlation in Delhi",
     "plot_correlation", "plot", "plot_correlation"),
    ("Scatter plot of rainfall vs PM2.5",
     "plot_correlation", "plot", "plot_correlation"),
    ("Correlation chart between ozone and temperature",
     "plot_correlation", "plot", "plot_correlation"),
    ("Plot PM10 vs NO2 in Mumbai",
     "plot_correlation", "plot", "plot_correlation"),
    ("Show wind speed vs PM10 correlation scatter",
     "plot_correlation", "plot", "plot_correlation"),
    ("Chart solar radiation vs ozone",
     "plot_correlation", "plot", "plot_correlation"),

    # ────────────── plot_ncap  (~5) ──────────────
    ("Plot NCAP funding by city",
     "plot_ncap", "plot", "plot_ncap"),
    ("Chart NCAP vs non-NCAP city pollution trends",
     "plot_ncap", "plot", "plot_ncap"),
    ("Show NCAP funding comparison chart",
     "plot_ncap", "plot", "plot_ncap"),
    ("Visualize PM2.5 reduction in NCAP vs non-NCAP cities",
     "plot_ncap", "plot", "plot_ncap"),
    ("Graph NCAP funding distribution",
     "plot_ncap", "plot", "plot_ncap"),

    # ────────────── plot_met_impact  (~10) ──────────────
    ("How does wind speed affect PM2.5? Show a chart",
     "plot_met_impact", "plot", "plot_met_impact"),
    ("Plot impact of temperature on PM10",
     "plot_met_impact", "plot", "plot_met_impact"),
    ("Show how humidity affects pollution in a chart",
     "plot_met_impact", "plot", "plot_met_impact"),
    ("Chart the effect of rainfall on PM2.5",
     "plot_met_impact", "plot", "plot_met_impact"),
    ("Does wind reduce PM2.5 in Delhi? Show a plot",
     "plot_met_impact", "plot", "plot_met_impact"),
    ("Impact of temperature on ozone levels chart",
     "plot_met_impact", "plot", "plot_met_impact"),
    ("How does wind speed impact PM10? Visualize it",
     "plot_met_impact", "plot", "plot_met_impact"),
    ("Show rainfall effect on PM2.5 in Mumbai chart",
     "plot_met_impact", "plot", "plot_met_impact"),
    ("Plot wind speed impact on NO2",
     "plot_met_impact", "plot", "plot_met_impact"),
    ("Visualize effect of humidity on PM2.5",
     "plot_met_impact", "plot", "plot_met_impact"),

    # ────────────── map_pollution  (~5) ──────────────
    ("Show PM2.5 levels across India on a map",
     "map_pollution", "plot", "map_pollution"),
    ("Map of most polluted cities in 2023",
     "map_pollution", "plot", "map_pollution"),
    ("Show top 20 polluted cities on a map for PM10",
     "map_pollution", "plot", "map_pollution"),
    ("Geographic distribution of pollution levels",
     "map_pollution", "plot", "map_pollution"),
    ("Show pollution map for Maharashtra",
     "map_pollution", "plot", "map_pollution"),

    # ────────────── map_change  (~5) ──────────────
    ("Map pollution change from 2020 to 2023",
     "map_change", "plot", "map_change"),
    ("Show which cities improved on a map",
     "map_change", "plot", "map_change"),
    ("Map of PM2.5 change between 2019 and 2024",
     "map_change", "plot", "map_change"),
    ("Geographic change in pollution over the years",
     "map_change", "plot", "map_change"),
    ("Map showing where air quality improved or worsened",
     "map_change", "plot", "map_change"),

    # ────────────── fallback / edge cases  (~5) ──────────────
    ("What is the standard deviation of NO2 on Tuesdays in coastal cities?",
     None, "fallback", "fallback"),
    ("Calculate the coefficient of variation of PM2.5 for each city",
     None, "fallback", "fallback"),
    ("How many monitoring stations are there per state?",
     None, "fallback", "fallback"),
    ("What is the population-weighted PM2.5 for each state?",
     None, "fallback", "fallback"),
    ("Compute the 90th percentile of PM2.5 for Delhi in winter",
     None, "fallback", "fallback"),
]


# ── Load data ───────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv("AQ_met_data.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    ncap_df = pd.read_csv("ncap_funding_data.csv")
    return df, ncap_df


# ── FC-only benchmark ──────────────────────────────────────────────────────

def run_fc(client, model_id, question):
    """Call Gemini FC and return (function_name | None, api_time_s)."""
    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=question,
            config=genai.types.GenerateContentConfig(
                system_instruction=FC_SYSTEM_PROMPT,
                tools=[analysis_tools],
                temperature=0,
            ),
        )
    except Exception as e:
        return None, time.time() - t0, str(e)
    api_time = time.time() - t0

    fc_name = None
    if resp.candidates and resp.candidates[0].content.parts:
        for part in resp.candidates[0].content.parts:
            if part.function_call and part.function_call.name:
                fc_name = part.function_call.name
                break

    return fc_name, api_time, None


def run_fc_exec(client, model_id, question, df, ncap_df):
    """Call FC + dispatch, return (function_name, api_time, exec_time, result_type, error)."""
    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=question,
            config=genai.types.GenerateContentConfig(
                system_instruction=FC_SYSTEM_PROMPT,
                tools=[analysis_tools],
                temperature=0,
            ),
        )
    except Exception as e:
        return None, time.time() - t0, 0, "error", str(e)
    api_time = time.time() - t0

    fc_part = None
    if resp.candidates and resp.candidates[0].content.parts:
        for part in resp.candidates[0].content.parts:
            if part.function_call and part.function_call.name:
                fc_part = part.function_call
                break

    if fc_part is None:
        return None, api_time, 0, "no_fc", None

    t1 = time.time()
    try:
        result = dispatch(fc_part, df, ncap_df)
        exec_time = time.time() - t1
    except Exception as e:
        return fc_part.name, api_time, time.time() - t1, "exec_error", str(e)

    if isinstance(result, pd.DataFrame):
        rtype = "table"
    elif isinstance(result, go.Figure):
        rtype = "plot"
    elif isinstance(result, str) and result.endswith((".png", ".jpg", ".jpeg")):
        rtype = "plot"
    else:
        rtype = "text"

    return fc_part.name, api_time, exec_time, rtype, None


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VayuChat FC test harness")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="Model key (default: gemini-2.0-flash)")
    parser.add_argument("--full", action="store_true",
                        help="Also dispatch functions (slower, generates plots)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just print question set, no API calls")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only run questions whose expected function contains this string")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run first N questions only")
    parser.add_argument("--output", type=str, default="test_results.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    # Filter questions
    questions = QUESTIONS
    if args.filter:
        questions = [(q, ef, eo, cat) for q, ef, eo, cat in questions
                     if args.filter in (ef or "") or args.filter in cat]
    if args.limit:
        questions = questions[:args.limit]

    print(f"Questions: {len(questions)}  |  Model: {args.model}  |  Mode: {'dry-run' if args.dry_run else ('full' if args.full else 'fc-only')}")
    print("=" * 80)

    # Dry run
    if args.dry_run:
        cat_counts = defaultdict(int)
        for i, (q, ef, eo, cat) in enumerate(questions, 1):
            cat_counts[cat] += 1
            print(f"  {i:3d}. [{cat:20s}] {q}")
        print("=" * 80)
        print("Category distribution:")
        for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat:20s}: {n}")
        print(f"  {'TOTAL':20s}: {len(questions)}")
        return

    # Load
    model_id = MODELS.get(args.model)
    if model_id is None:
        print(f"Unknown model: {args.model}. Available: {', '.join(MODELS)}")
        sys.exit(1)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not set")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    df, ncap_df = None, None
    if args.full:
        df, ncap_df = load_data()

    # Run
    results = []
    for i, (question, expected_fn, expected_type, category) in enumerate(questions, 1):
        print(f"[{i:3d}/{len(questions)}] {question[:70]}...", end="", flush=True)

        if args.full and df is not None:
            fc_name, api_time, exec_time, result_type, error = run_fc_exec(
                client, model_id, question, df, ncap_df
            )
        else:
            fc_name, api_time, error = run_fc(client, model_id, question)
            exec_time = 0
            result_type = None

        # Determine correctness (expected_fn can be str, list, or None)
        if expected_fn is None:
            # Fallback expected — correct if NO function call
            correct = fc_name is None
        elif isinstance(expected_fn, list):
            correct = fc_name in expected_fn
        else:
            correct = fc_name == expected_fn

        # Determine type correctness (only in full mode)
        type_correct = None
        if args.full and result_type and expected_type != "fallback":
            type_correct = result_type == expected_type

        status = "OK" if correct else "MISS"
        fn_display = fc_name or "(none)"
        print(f"  {api_time:.2f}s  {fn_display:20s} [{status}]"
              + (f"  exec={exec_time:.2f}s  type={result_type}" if args.full else "")
              + (f"  ERR: {error}" if error else ""))

        expected_display = expected_fn if isinstance(expected_fn, str) else "|".join(expected_fn) if expected_fn else "(fallback)"
        results.append({
            "question": question,
            "category": category,
            "expected_function": expected_display,
            "actual_function": fc_name or "(none)",
            "correct": correct,
            "expected_type": expected_type,
            "actual_type": result_type or "",
            "type_correct": type_correct if type_correct is not None else "",
            "api_time_s": round(api_time, 3),
            "exec_time_s": round(exec_time, 3) if args.full else "",
            "total_time_s": round(api_time + exec_time, 3),
            "error": error or "",
        })

        # Gentle rate limiting to avoid quota issues
        if i < len(questions):
            time.sleep(0.3)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults written to {args.output}")

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    print(f"\nTool Selection Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")

    # Per-category accuracy
    print(f"\n{'Category':24s} {'Correct':>8s} {'Total':>6s} {'Accuracy':>9s} {'Avg API':>8s}")
    print("-" * 60)
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0, "api_times": []})
    for r in results:
        cat = r["category"]
        cat_stats[cat]["total"] += 1
        cat_stats[cat]["api_times"].append(r["api_time_s"])
        if r["correct"]:
            cat_stats[cat]["correct"] += 1

    for cat in sorted(cat_stats, key=lambda c: -cat_stats[c]["total"]):
        s = cat_stats[cat]
        acc = 100 * s["correct"] / s["total"] if s["total"] else 0
        avg_api = sum(s["api_times"]) / len(s["api_times"])
        print(f"  {cat:22s} {s['correct']:>6d} {s['total']:>6d} {acc:>8.1f}% {avg_api:>7.2f}s")

    # Timing summary
    api_times = [r["api_time_s"] for r in results]
    print(f"\nAPI Timing (FC path):")
    print(f"  Mean:   {sum(api_times)/len(api_times):.2f}s")
    print(f"  Median: {sorted(api_times)[len(api_times)//2]:.2f}s")
    print(f"  Min:    {min(api_times):.2f}s")
    print(f"  Max:    {max(api_times):.2f}s")
    print(f"  p95:    {sorted(api_times)[int(len(api_times)*0.95)]:.2f}s")

    if args.full:
        exec_times = [r["exec_time_s"] for r in results if isinstance(r["exec_time_s"], (int, float)) and r["exec_time_s"] > 0]
        if exec_times:
            print(f"\nExecution Timing:")
            print(f"  Mean:   {sum(exec_times)/len(exec_times):.3f}s")
            print(f"  Max:    {max(exec_times):.3f}s")

        type_results = [r for r in results if r["type_correct"] != ""]
        if type_results:
            type_correct = sum(1 for r in type_results if r["type_correct"])
            print(f"\nOutput Type Accuracy: {type_correct}/{len(type_results)} ({100*type_correct/len(type_results):.1f}%)")

    # Mismatches
    mismatches = [r for r in results if not r["correct"]]
    if mismatches:
        print(f"\nMISMATCHES ({len(mismatches)}):")
        for r in mismatches:
            print(f"  Q: {r['question'][:70]}")
            print(f"     expected={r['expected_function']}, got={r['actual_function']}")
    else:
        print("\nNo mismatches — perfect tool selection!")

    # Errors
    errors = [r for r in results if r["error"]]
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for r in errors:
            print(f"  Q: {r['question'][:70]}")
            print(f"     {r['error'][:100]}")

    print()


if __name__ == "__main__":
    main()
