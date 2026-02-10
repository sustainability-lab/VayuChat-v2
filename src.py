import os
import pandas as pd
from typing import Tuple
from PIL import Image
from dotenv import load_dotenv
from google import genai
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
from datetime import datetime
from huggingface_hub import HfApi
import uuid
import threading
from analysis_tools import analysis_tools, FC_SYSTEM_PROMPT, dispatch

# Load environment variables once at startup
load_dotenv(override=True)

# Get API keys
hf_token = os.getenv("HF_TOKEN")
gemini_token = os.getenv("GOOGLE_API_KEY")

models = {
    "gemini-3-flash": "gemini-3-flash-preview",
    "gemini-3-pro": "gemini-3-pro-preview",
    "gemini-2.5-flash": "gemini-2.5-flash",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "gemini-2.0-flash-lite": "gemini-2.0-flash-lite",
}

# --- Cached resources (loaded once, reused across queries) ---
_client = None
_cached_data = None
_system_prompt = None

def _get_client():
    """Reuse a single Gemini client across queries."""
    global _client
    if _client is None and gemini_token:
        _client = genai.Client(api_key=gemini_token)
    return _client

def _get_data():
    """Load CSVs once and cache them."""
    global _cached_data
    if _cached_data is None:
        df = pd.read_csv("AQ_met_data.csv")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        states_df = pd.read_csv("states_data.csv")
        ncap_df = pd.read_csv("ncap_funding_data.csv")
        # Pre-build the dtypes comment block for the template
        nl = "\n"
        dtype_comments = {
            "df": nl.join("# " + x for x in str(df.dtypes).split(nl)),
            "states_df": nl.join("# " + x for x in str(states_df.dtypes).split(nl)),
            "ncap_df": nl.join("# " + x for x in str(ncap_df.dtypes).split(nl)),
        }
        _cached_data = (df, states_df, ncap_df, dtype_comments)
    return _cached_data

def _get_system_prompt():
    """Read system prompt once and cache it."""
    global _system_prompt
    if _system_prompt is None:
        with open("new_system_prompt.txt", "r", encoding="utf-8") as f:
            _system_prompt = f.read().strip()
    return _system_prompt

# Set matplotlib rcParams once at import time
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'figure.figsize': [9, 6],
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

def _log_interaction_sync(user_query, model_name, response_content, generated_code, execution_time, error_message=None, is_image=False):
    """Actual logging work — runs in a background thread."""
    try:
        if not hf_token or hf_token.strip() == "":
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": str(uuid.uuid4()),
            "user_query": user_query,
            "model_name": model_name,
            "response_content": str(response_content),
            "generated_code": generated_code or "",
            "execution_time_seconds": execution_time,
            "error_message": error_message or "",
            "is_image_output": is_image,
            "success": error_message is None
        }

        log_df = pd.DataFrame([log_entry])
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        filename = f"interaction_log_{timestamp_str}_{random_id}.parquet"
        local_path = f"/tmp/{filename}"
        log_df.to_parquet(local_path, index=False)

        api = HfApi(token=hf_token)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"data/{filename}",
            repo_id="SustainabilityLabIITGN/VayuChat_logs",
            repo_type="dataset",
        )

        if os.path.exists(local_path):
            os.remove(local_path)

    except Exception as e:
        print(f"Error logging interaction: {e}")

def log_interaction(user_query, model_name, response_content, generated_code, execution_time, error_message=None, is_image=False):
    """Fire-and-forget: log in a background thread so the user isn't blocked."""
    t = threading.Thread(
        target=_log_interaction_sync,
        args=(user_query, model_name, response_content, generated_code, execution_time),
        kwargs={"error_message": error_message, "is_image": is_image},
        daemon=True,
    )
    t.start()

def preprocess_and_load_df(path: str) -> pd.DataFrame:
    """Load and preprocess the dataframe"""
    try:
        df = pd.read_csv(path)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        return df
    except Exception as e:
        raise Exception(f"Error loading dataframe: {e}")


def get_from_user(prompt):
    """Format user prompt"""
    return {"role": "user", "content": prompt}


def ask_question(model_name, question):
    """Ask question with comprehensive error handling and logging"""
    start_time = datetime.now()
    # ------------------------
    # Helper functions
    # ------------------------
    def make_error_response(msg, log_msg, content=None):
        """Build error response + log it"""
        execution_time = (datetime.now() - start_time).total_seconds()
        log_interaction(
            user_query=question,
            model_name=model_name,
            response_content=content or msg,
            generated_code="",
            execution_time=execution_time,
            error_message=log_msg,
            is_image=False
        )
        return {
            "role": "assistant",
            "content": content or msg,
            "gen_code": "",
            "ex_code": "",
            "last_prompt": question,
            "error": log_msg
        }
    def validate_api_token(token, token_name, msg_if_missing):
        """Check for missing/empty API tokens"""
        if not token or token.strip() == "":
            return make_error_response(
                msg="Missing or empty API token",
                log_msg="Missing or empty API token",
                content=msg_if_missing
            )
        return None  # OK
    def run_safe_exec(full_code, df=None, extra_globals=None):
        """Safely execute generated code and handle errors"""
        local_vars = {}
        # Apply style once per exec (cheap if already active)
        try:
            plt.style.use('vayuchat.mplstyle')
        except OSError:
            pass
        global_vars = {
            'pd': pd, 'plt': plt, 'os': os,
            'sns': __import__('seaborn'),
            'uuid': __import__('uuid'),
            'calendar': __import__('calendar'),
            'np': __import__('numpy'),
            'df': df,
            'st': __import__('streamlit')
        }

        if extra_globals:
            global_vars.update(extra_globals)

        try:
            exec(full_code, global_vars, local_vars)
            return (
                local_vars.get('answer', "Code executed but no result was saved in 'answer' variable"),
                None
            )
        except Exception as code_error:
            return None, str(code_error)

    # ------------------------
    # Step 1: Validate token & get client
    # ------------------------
    token_error = validate_api_token(
        gemini_token,
        "GOOGLE_API_KEY",
        "Gemini API token not available or empty. Please set GOOGLE_API_KEY in your environment variable."
    )
    if token_error:
        return token_error

    client = _get_client()
    if client is None:
        return make_error_response("API Connection Error", "Failed to create Gemini client")

    # ------------------------
    # Step 2: Load cached data
    # ------------------------
    if not os.path.exists("AQ_met_data.csv"):
        return make_error_response(
            msg="Data file not found",
            log_msg="Data file not found",
            content="AQ_met_data.csv file not found. Please ensure the data file is in the correct location."
        )

    t0 = datetime.now()
    df, states_df, ncap_df, dtype_comments = _get_data()
    system_prompt = _get_system_prompt()
    q_short = question.strip()[:80]
    print(f"[TIMING] data+prompt load: {(datetime.now()-t0).total_seconds():.3f}s  q={q_short!r}")

    # ================================================================
    # PATH A: Native Function Calling (fast path)
    # ================================================================
    t_fc = datetime.now()
    fc_part = None
    try:
        fc_response = client.models.generate_content(
            model=models[model_name],
            contents=question,
            config=genai.types.GenerateContentConfig(
                system_instruction=FC_SYSTEM_PROMPT,
                tools=[analysis_tools],
                temperature=0,
            ),
        )
        print(f"[TIMING] FC API call: {(datetime.now()-t_fc).total_seconds():.3f}s  q={q_short!r}")

        # Check if the model returned a function call
        if fc_response.candidates and fc_response.candidates[0].content.parts:
            for part in fc_response.candidates[0].content.parts:
                if part.function_call and part.function_call.name:
                    fc_part = part.function_call
                    break

        if fc_part is None:
            fc_text = ""
            try:
                fc_text = fc_response.text or ""
            except Exception:
                pass
            print(f"[FC] No function call for q={q_short!r}, falling back to code-gen." + (f" (text: {fc_text[:80]}...)" if fc_text else ""))

    except Exception as fc_err:
        print(f"[FC] API call failed for q={q_short!r} ({fc_err}), falling back to code-gen.")

    # Dispatch the function call (outside the API try/except so errors surface clearly)
    if fc_part is not None:
        print(f"[FC] q={q_short!r} → {fc_part.name}({dict(fc_part.args) if fc_part.args else {}})")
        t_exec = datetime.now()
        fc_result = dispatch(fc_part, df, ncap_df)
        print(f"[TIMING] FC exec: {(datetime.now()-t_exec).total_seconds():.3f}s")

        execution_time = (datetime.now() - start_time).total_seconds()
        print(f"[TIMING] total (FC path): {execution_time:.3f}s  q={q_short!r}")

        is_image = isinstance(fc_result, str) and fc_result.endswith(('.png', '.jpg', '.jpeg'))
        is_plotly = isinstance(fc_result, go.Figure)
        fc_code_desc = f"[Function Call] {fc_part.name}({dict(fc_part.args) if fc_part.args else {}})"
        log_content = "[Interactive Map]" if is_plotly else str(fc_result)
        log_interaction(
            user_query=question,
            model_name=model_name,
            response_content=log_content,
            generated_code=fc_code_desc,
            execution_time=execution_time,
            error_message=None,
            is_image=is_image
        )
        return {
            "role": "assistant",
            "content": fc_result,
            "gen_code": fc_code_desc,
            "ex_code": fc_code_desc,
            "last_prompt": question,
            "error": None
        }

    # ================================================================
    # PATH B: Code-Gen Fallback (existing path)
    # ================================================================
    template = f"""```python
# All imports and data are pre-loaded. Do NOT call pd.read_csv() or any file reads for these.
# Available: pd, plt, sns, st, np, uuid, calendar, os
# plt style 'vayuchat.mplstyle' is already applied.
#
# df — air quality DataFrame (daily, 2017–2024, India). Columns:
{dtype_comments["df"]}
# states_df — state-wise population, area, union territory flag. Columns:
{dtype_comments["states_df"]}
# ncap_df — NCAP funding by city, 2019–2022. Columns:
{dtype_comments["ncap_df"]}
#
# Question: {question.strip()}
# Write code to answer the question. Store the result in 'answer'.
# If creating a plot, save it with a unique filename and store the filename in 'answer'.
```"""

    user_content = f"Complete the following code to answer the user's question: \n{template}"

    # Call Gemini for code generation
    t1 = datetime.now()
    try:
        response = client.models.generate_content(
            model=models[model_name],
            contents=user_content,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,
            ),
        )
        answer = response.text
    except Exception as e:
        return make_error_response(f"Error: {e}", str(e))
    print(f"[TIMING] Code-gen API call: {(datetime.now()-t1).total_seconds():.3f}s  q={q_short!r}")

    # Extract code
    if "```python" in answer:
        code_part = answer.split("```python")[1].split("```")[0]
    elif "```" in answer:
        code_part = answer.split("```")[1].split("```")[0]
    else:
        code_part = answer

    full_code = f"""
{template.split("```python")[1].split("```")[0]}
{code_part}
"""
    t2 = datetime.now()
    answer_result, code_error = run_safe_exec(full_code, df, extra_globals={'states_df': states_df, 'ncap_df': ncap_df})
    print(f"[TIMING] code exec: {(datetime.now()-t2).total_seconds():.3f}s")

    execution_time = (datetime.now() - start_time).total_seconds()
    print(f"[TIMING] total (code-gen path): {execution_time:.3f}s  q={q_short!r}")
    if code_error:
        msg = "I encountered an error while analyzing your data. "
        if "syntax" in code_error.lower():
            msg += "There was a syntax error in the generated code. Please try rephrasing your question."
        elif "not defined" in code_error.lower():
            msg += "Variable naming error occurred. Please try asking the question again."
        elif "division by zero" in code_error.lower():
            msg += "Calculation involved division by zero, possibly due to missing data."
        elif "no data" in code_error.lower() or "empty" in code_error.lower():
            msg += "No relevant data was found for your query."
        else:
            msg += f"Technical error: {code_error}"

        msg += "\n\n💡 **Suggestions:**\n- Try rephrasing your question\n- Use simpler terms\n- Check if the data exists for your specified criteria"

        log_interaction(
            user_query=question,
            model_name=model_name,
            response_content=msg,
            generated_code=full_code,
            execution_time=execution_time,
            error_message=code_error,
            is_image=False
        )
        return {
            "role": "assistant",
            "content": msg,
            "gen_code": full_code,
            "ex_code": full_code,
            "last_prompt": question,
            "error": code_error
        }

    # Success logging
    is_image = isinstance(answer_result, str) and answer_result.endswith(('.png', '.jpg', '.jpeg'))
    log_interaction(
        user_query=question,
        model_name=model_name,
        response_content=str(answer_result),
        generated_code=full_code,
        execution_time=execution_time,
        error_message=None,
        is_image=is_image
    )

    return {
        "role": "assistant",
        "content": answer_result,
        "gen_code": full_code,
        "ex_code": full_code,
        "last_prompt": question,
        "error": None
    }
