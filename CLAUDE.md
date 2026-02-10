# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VayuChat is a Streamlit-based conversational AI application for air quality data analysis. It provides an interactive chat interface where users can ask questions about PM2.5 and PM10 pollution data through natural language, and receive responses including visualizations and data insights.

## Architecture

The application follows a two-file architecture:

- **app.py**: Main Streamlit application with UI components, chat interface, and user interaction handling
- **src.py**: Core data processing logic, LLM integration, and code generation/execution engine

Key architectural patterns:
- **Code Generation Pipeline**: User questions are converted to executable Python code via LLM prompting, then executed dynamically
- **Multi-LLM Support**: Supports both Groq (LLaMA models) and Google Gemini models through LangChain
- **Session Management**: Uses Streamlit session state for chat history and user interactions
- **Feedback Loop**: Comprehensive logging and feedback collection to HuggingFace datasets

## Development Commands

### Run the Application
```bash
streamlit run app.py
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Environment Setup
Create a `.env` file with the following variables:
```bash
GROQ_API_KEY=your_groq_api_key_here
GEMINI_TOKEN=your_google_gemini_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional, for logging
```

## Data Requirements

- **Data.csv**: Must contain columns: `Timestamp`, `station`, `PM2.5`, `PM10`, `address`, `city`, `latitude`, `longitude`, `state`
- **IITGN_Logo.png**: Logo image for the sidebar
- **questions.txt**: Pre-defined quick prompt questions (optional)
- **system_prompt.txt**: Contains specific instructions for the LLM code generation

## Code Generation System

The application uses a unique code generation approach in `src.py`:

1. **Template-based Code Generation**: User questions are embedded into a Python code template that includes data loading and analysis patterns
2. **Dynamic Execution**: Generated code is executed in a controlled environment with pandas, matplotlib, and other libraries available
3. **Result Handling**: Results are stored in an `answer` variable and can be either text/numbers or plot file paths
4. **Error Recovery**: Comprehensive error handling with logging to HuggingFace datasets

## Key Functions (src.py)

- `ask_question()`: Main entry point for processing user queries
- `preprocess_and_load_df()`: Data loading and preprocessing
- `load_agent()` / `load_smart_df()`: LLM agent initialization
- `log_interaction()`: Interaction logging to HuggingFace
- `upload_feedback()`: User feedback collection (in app.py)

## Model Configuration

Available models are defined in both files:
- Groq models: LLaMA 3.1, LLaMA 3.3, LLaMA 4 variants, DeepSeek-R1, GPT-OSS
- Google models: Gemini 1.5 Pro

## Plotting Guidelines

When generating visualization code, the system follows specific guidelines from `system_prompt.txt`:
- Include India (60 µg/m³) and WHO (15 µg/m³) guidelines for PM2.5
- Include India (100 µg/m³) and WHO (50 µg/m³) guidelines for PM10
- Use tight layout and 45-degree rotated x-axis labels
- Save plots with unique filenames using UUID
- Use 'Reds' colormap for air quality visualizations
- Round floating point numbers to 2 decimal places
- Always report units (µg/m³) and include standard deviation/error for aggregations

## Logging and Feedback

- All interactions are logged to `SustainabilityLabIITGN/VayuChat_logs` HuggingFace dataset
- User feedback is collected and stored in `SustainabilityLabIITGN/VayuChat_Feedback` dataset
- Session tracking via UUID for analytics