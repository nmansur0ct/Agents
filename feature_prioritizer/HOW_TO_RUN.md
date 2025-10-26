# How to Run the Feature Prioritization Assistant

## Quick Start Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Optional: OpenAI API key for LLM enhancements

### Step 1: Install Dependencies

```bash
cd /Users/n0m08hp/Agents/feature_prioritizer
pip install -r requirements.txt
```

**Key dependencies installed:**
- `pydantic>=2.8.2` - Data validation
- `click>=8.0.0` - CLI framework
- `pandas>=1.5.0` - Data processing
- `langgraph>=0.2.34` - Workflow orchestration
- `openai>=1.47.0` - OpenAI API integration
- `python-dotenv>=1.0.1` - Environment variable loading

### Step 2: Configure Environment (Optional for LLM)

The system works perfectly without LLM, but for enhanced analysis:

1. **Set up OpenAI API key** (if you want LLM enhancements):
   ```bash
   # Option 1: Use existing .env file (already configured)
   # The .env file at /Users/n0m08hp/Agents/.env contains:
   # OPENAI_API_AGENT_KEY=your_api_key_here
   
   # Option 2: Set environment variable directly
   export OPENAI_API_KEY=your_api_key_here
   ```

2. **Verify environment loading**:
   ```bash
   python -c "from llm_utils import call_openai_json; print('Environment loaded successfully')"
   ```
   Should show: `✓ Loaded environment variables from /Users/n0m08hp/Agents/.env`

## Running the Application

### Basic Usage (Without LLM)

#### 1. Using JSON File Input
```bash
python run.py --file samples/features.json --metric RICE
```

#### 2. Using CSV File Input
```bash
python run.py --file samples/features.csv --metric ICE
```

#### 3. Using JSON String Input
```bash
python run.py --json '[{"name": "Test Feature", "description": "A test feature for checkout optimization"}]' --metric RICE
```

### Enhanced Usage (With LLM)

#### 1. Basic LLM Enhancement
```bash
python run.py --file samples/features.json --llm --verbose
```

#### 2. LLM with Specific Model
```bash
python run.py --file samples/features.json --llm --model gpt-4o-mini --verbose
```

#### 3. LLM with Custom API Key Environment
```bash
python run.py --file samples/features.json --llm --keyenv CUSTOM_OPENAI_KEY --verbose
```

### Output Options

#### 1. JSON Output File
```bash
python run.py --file samples/features.json --output results.json
```

#### 2. CSV Output File
```bash
python run.py --file samples/features.json --csv-output results.csv
```

#### 3. Detailed Results (All Processing Stages)
```bash
python run.py --file samples/features.json --detailed --output detailed_results.json
```

## Command Line Options Reference

| Option | Short | Description | Example |
|--------|-------|-------------|---------|
| `--file` | `-f` | Path to JSON or CSV file | `--file features.json` |
| `--json` | `-j` | JSON string input | `--json '[{"name":"Feature1",...}]'` |
| `--metric` | `-m` | Scoring policy | `--metric RICE` (RICE/ICE/ImpactMinusEffort) |
| `--output` | `-o` | JSON output file | `--output results.json` |
| `--csv-output` | `-c` | CSV output file | `--csv-output results.csv` |
| `--detailed` | `-d` | Include all processing stages | `--detailed` |
| `--verbose` | `-v` | Show processing details | `--verbose` |
| `--llm` | | Enable LLM enhancements | `--llm` |
| `--model` | | OpenAI model to use | `--model gpt-3.5-turbo` |
| `--keyenv` | | API key environment variable | `--keyenv OPENAI_API_KEY` |

## Sample Data

The project includes sample data files in the `samples/` directory:

### 1. `samples/features.json`
Contains 15 sample features with various descriptions for testing.

### 2. `samples/features.csv`
Same features in CSV format for spreadsheet compatibility.

### View Sample Data
```bash
# View JSON sample
cat samples/features.json | head -20

# View CSV sample
head -10 samples/features.csv
```

## Example Workflows

### 1. Quick Analysis (Keyword-based)
```bash
# Fast analysis using keyword heuristics
python run.py --file samples/features.json --metric RICE --verbose
```
**Output**: Prioritized features with RICE scoring in ~1 second

### 2. Enhanced Analysis (With LLM)
```bash
# Intelligent analysis with LLM insights
python run.py --file samples/features.json --llm --verbose
```
**Output**: Enhanced factor analysis and business rationales (requires API key)

### 3. Comprehensive Report
```bash
# Generate detailed report with all stages
python run.py --file samples/features.json --detailed --output full_report.json --csv-output summary.csv
```
**Output**: Complete analysis with JSON and CSV outputs

### 4. Custom Feature Analysis
```bash
# Analyze your own features
python run.py --json '[
  {
    "name": "Mobile Payment Integration",
    "description": "Add Apple Pay and Google Pay support to checkout process"
  },
  {
    "name": "Advanced Search Filters", 
    "description": "Machine learning powered search with personalized filtering options"
  }
]' --llm --metric ICE --verbose
```

## Understanding the Output

### Basic Output Format
```json
{
  "prioritized_features": [
    {
      "name": "Feature Name",
      "impact": 0.85,
      "effort": 0.425,
      "score": 2.0,
      "rationale": "High reach; Revenue impact; Scored using RICE"
    }
  ]
}
```

### Detailed Output Format (with --detailed)
```json
{
  "extracted_features": [...],
  "scored_features": [...],
  "prioritized_features": [...],
  "config": {...},
  "processing_notes": [...]
}
```

## LLM Integration Features

When using `--llm`, you'll see detailed logging:

```
✓ Loaded environment variables from /Users/n0m08hp/Agents/.env
🔍 LLM Factor Analysis requested for feature: 'Mobile Payment Integration...'
🤖 GPT LLM invoked for analysis using model: gpt-3.5-turbo
   📝 Prompt length: 1310 characters
   🎛️  Temperature: 0.3
   🌐 Making OpenAI API call...
   ✅ LLM Analysis complete - received 8 fields
   ✨ LLM Factor Analysis completed successfully
      💡 High revenue potential due to payment optimization
      💡 Moderate engineering effort for API integration
```

### LLM Logging Icons Guide
- 🔍 Factor Analysis Request
- 📝 Rationale Generation  
- 🤖 LLM API Call
- ✅ Success
- ⚠️ Fallback to keyword analysis
- ❌ Error (with details)
- 💡 Key insights from LLM

## Testing and Validation

### 1. Test Basic Functionality
```bash
python run.py --help
```

### 2. Test Sample Data
```bash
python run.py --file samples/features.json --verbose
```

### 3. Test LLM Integration
```bash
python demo_llm_logging.py
```

### 4. Test Environment Loading
```bash
python -c "from config import Config; print('✓ Config loaded'); from llm_utils import call_openai_json; print('✓ LLM utils loaded')"
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Import Errors
```bash
# Issue: ModuleNotFoundError
# Solution: Install dependencies
pip install -r requirements.txt
```

#### 2. API Key Issues
```bash
# Issue: Missing API key
# Solution: Check environment variables
python -c "import os; print(f'API Key: {os.getenv(\"OPENAI_API_AGENT_KEY\", \"Not found\")}')"
```

#### 3. File Not Found
```bash
# Issue: Input file doesn't exist
# Solution: Use absolute path or verify file location
ls -la samples/features.json
```

#### 4. LLM Connection Issues
- The system gracefully falls back to keyword analysis
- Check internet connection and API key validity
- Verify API quota and billing status

### Performance Notes

- **Without LLM**: Processing is nearly instantaneous (< 1 second for 15 features)
- **With LLM**: Processing takes 1-3 seconds per feature due to API calls
- **Timeout**: LLM calls timeout after 30 seconds (configurable)
- **Fallback**: System never hangs or fails completely

## Advanced Usage

### 1. Batch Processing Multiple Files
```bash
# Process multiple feature sets
for file in samples/*.json; do
  echo "Processing $file"
  python run.py --file "$file" --llm --output "results_$(basename $file)"
done
```

### 2. Custom Configuration
```python
# Python script for custom configuration
from config import Config
from models import RawFeature
from graph import FeaturePrioritizationGraph

# Create custom config
config = Config.llm_enabled_config()
config.llm_model = "gpt-4"
config.llm_temperature = 0.1

# Process features
features = [RawFeature(name="Custom Feature", description="...")]
graph = FeaturePrioritizationGraph(config)
results = graph.get_prioritized_features(features)
```

### 3. Integration with Other Tools
```bash
# Pipe results to other tools
python run.py --file features.json --csv-output /dev/stdout | sort -k3 -nr
```

## Next Steps

1. **Start Simple**: Use basic keyword analysis to understand the system
2. **Add LLM**: Configure API key and enable LLM for enhanced insights
3. **Customize**: Adjust scoring policies and weights for your domain
4. **Integrate**: Use the CLI or Python API in your workflows
5. **Scale**: Process larger feature sets and automate decision-making

For more detailed information, see:
- `LLM_Enhancement_Design.md` - Technical architecture
- `ENV_and_Logging_Implementation.md` - Environment and logging details
- `LLM_Implementation_Summary.md` - Complete implementation overview