# 🌤️ Weather-Based Recommendation Agent

An intelligent AI agent that provides context-aware, actionable weather recommendations using Agentic AI principles. Unlike traditional weather apps that only display raw data, this agent focuses on decision support—helping you make informed choices about clothing, travel, and activities.

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="https://groq.com/"><img src="https://img.shields.io/badge/Groq-FF6B6B?style=for-the-badge&logo=ai&logoColor=white" alt="Groq"></a>
  <a href="https://www.langchain.com/"><img src="https://img.shields.io/badge/LangChain-121212?style=for-the-badge&logo=chainlink&logoColor=white" alt="LangChain"></a>
  <a href="https://github.com/langchain-ai/langgraph"><img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=for-the-badge&logo=graph&logoColor=white" alt="LangGraph"></a>
  <a href="https://www.gradio.app/"><img src="https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white" alt="Gradio"></a>
  <a href="https://python-dotenv.readthedocs.io/"><img src="https://img.shields.io/badge/.ENV-ECD53F?style=for-the-badge&logo=dotenv&logoColor=black" alt="dotenv"></a>
  <img src="https://img.shields.io/badge/AI-Agentic%20System-blueviolet?style=for-the-badge" alt="Agentic AI">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge" alt="Status">
</p>


## 🎯 Problem Statement

Most weather platforms present information such as temperature, rainfall, and wind speed in isolation, requiring users to interpret the data themselves. However, users typically ask questions like:

- *"What clothes should I pack?"*
- *"Is it safe to travel?"*
- *"Will weather affect my plans?"*

**This project bridges that gap** by combining live weather data with intelligent reasoning to generate meaningful, personalized recommendations.

## ✨ Features

### 🧠 Intelligent Capabilities
- **Natural Language Understanding**: Ask questions in plain English
- **Location Extraction**: Automatically identifies locations from your query
- **Real-time Weather Data**: Fetches current conditions and 3-day forecasts
- **Context-Aware Recommendations**: Tailored advice based on your intent

### 📋 Recommendation Types
- 👔 **Clothing Suggestions**: Based on temperature, wind, rain, and UV index
- 🚗 **Travel Safety**: Assess weather-related risks and precautions
- 📅 **Activity Planning**: Best times and conditions for outdoor activities
- 💊 **Health Considerations**: UV protection, hydration, and comfort tips
- ⚠️ **Weather Alerts**: Real-time severe weather notifications

## 🏗️ System Architecture

The system follows an **agentic workflow** orchestrated using LangGraph:

```
┌─────────────────┐
│  User Query     │
│  (Natural Lang) │
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ 1. Extract Location │  ← LLM extracts geographic location
│    (LLM Parsing)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 2. Fetch Weather    │  ← WeatherAPI.com (Real-time + Forecast)
│    (API Call)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 3. Analyze Weather  │  ← Programmatic analysis of weather signals
│    (Logic Layer)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ 4. Generate Advice  │  ← LLM creates personalized recommendations
│    (LLM Reasoning)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Recommendations    │  → User-friendly, actionable advice
└─────────────────────┘
```

### State-Driven Execution with LangGraph

```python
State Flow:
├── extract_location (Node 1)
│   ├── Success → fetch_weather
│   └── Error → error_handler
├── fetch_weather (Node 2)
│   ├── Success → analyze
│   └── Error → error_handler
├── analyze (Node 3) → END
└── error_handler → END
```

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | Agent workflow and state management |
| **LLM** | Groq + LLaMA 3.3 70B | Fast inference for location extraction and recommendations |
| **Weather API** | WeatherAPI.com | Real-time weather data and forecasting |
| **UI Framework** | Gradio | User-friendly web interface |
| **Language** | Python 3.8+ | Implementation |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for API calls

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/weather-recommendation-agent.git
cd weather-recommendation-agent
```

### Step 2: Install Dependencies
```bash
pip install langchain-groq langgraph gradio requests
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Step 3: Get API Keys

#### Groq API Key (Free)
1. Visit [Groq Console](https://console.groq.com/)
2. Sign up for a free account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key (starts with `gsk_`)

#### WeatherAPI Key (Free)
1. Visit [WeatherAPI.com](https://www.weatherapi.com/)
2. Sign up for a free account
3. Get your API key from the dashboard
4. Free tier includes: 1M calls/month

### Step 4: Configure API Keys

**Option 1: Environment Variables (Recommended)**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export WEATHER_API_KEY="your_weather_api_key_here"
```

**Option 2: Enter in UI**
- Launch the app and enter keys directly in the interface

## 📖 Usage

### Running the Application

```bash
python weather_agent.py
```

The application will start and open at: `http://localhost:7860`

### Example Queries

#### Clothing Recommendations
```
"What should I wear in Paris tomorrow?"
"What clothes should I pack for a week in Tokyo?"
"Do I need a jacket in London today?"
```

#### Travel Planning
```
"Is it safe to travel to Mumbai next week?"
"What's the best time to visit Sydney for beach activities?"
"Will weather affect my road trip to Seattle?"
```

#### Activity Planning
```
"Should I bring an umbrella to New York?"
"Is it good weather for hiking in Denver this weekend?"
"Can I plan an outdoor event in Chicago on Saturday?"
```

### Sample Interaction

**User Query:**
> "What should I wear in London tomorrow?"

**Agent Response:**
```markdown
## 🌤️ Weather Overview for London, United Kingdom

**Current Conditions:** 12°C (54°F), Partly Cloudy
**Tomorrow's Forecast:** 8-14°C (46-57°F), Light Rain Expected

## 👔 Clothing Recommendations

1. **Layers are Essential**
   - Base layer: Long-sleeve shirt or light sweater
   - Mid layer: Cardigan or light fleece
   - Outer layer: Waterproof jacket or raincoat

2. **Bottom Wear**
   - Long pants or jeans recommended
   - Avoid shorts due to cooler temperatures

3. **Accessories**
   - ☂️ Umbrella (60% chance of rain)
   - 🧤 Light gloves optional for morning/evening
   - 👟 Waterproof or water-resistant shoes

## 🚗 Travel Considerations

- Roads may be slippery due to rain - drive carefully
- Public transport running normally
- Allow extra time for travel due to wet conditions

## 💡 Additional Tips

- UV Index: Low (2/10) - minimal sun protection needed
- Humidity: 75% - comfortable conditions
- Visibility: Good (10km)
```

## 🎨 UI Features

### Gradio Interface Components

1. **API Key Input**
   - Secure password fields for API keys
   - Option to use environment variables
   - Links to get API keys

2. **Query Input**
   - Large text area for natural language queries
   - Pre-populated example queries
   - Clear placeholder text

3. **Recommendations Output**
   - Markdown-formatted responses
   - Structured sections with headers
   - Weather alerts highlighted

4. **Example Queries**
   - One-click query examples
   - Demonstrates various use cases
   - Helps users understand capabilities

## 🔧 Configuration Options

### Customizing the LLM

Edit the `get_llm()` function to change model or parameters:

```python
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",  # Change model here
        temperature=0.3,  # Adjust creativity (0.0-1.0)
        max_tokens=1024   # Adjust response length
    )
```

**Available Groq Models:**
- `llama-3.3-70b-versatile` - Best quality (recommended)
- `llama-3.1-8b-instant` - Fastest
- `mixtral-8x7b-32768` - Large context window
- `gemma2-9b-it` - Efficient alternative

### Adjusting Weather Forecast Days

Modify the `fetch_weather_data()` function:

```python
params = {
    "key": WEATHER_API_KEY,
    "q": location,
    "days": 3,  # Change to 1-14 days
    "aqi": "no",
    "alerts": "yes"
}
```

## 📊 Agent Workflow Details

### Node 1: Location Extraction
```python
Input: "What should I wear in Paris tomorrow?"
Process: LLM analyzes query and extracts location
Output: location = "Paris"
```

### Node 2: Weather Data Fetch
```python
Input: location = "Paris"
Process: API call to WeatherAPI.com
Output: {
  "current": {...},
  "forecast": [...],
  "alerts": [...]
}
```

### Node 3: Analysis & Recommendations
```python
Input: user_query + weather_data
Process: LLM generates contextualized recommendations
Output: Structured markdown with advice
```

## 🔒 Error Handling

The agent includes robust error handling:

- **Missing API Keys**: Clear error messages with setup instructions
- **Invalid Location**: Prompts user to specify a location
- **API Failures**: Graceful degradation with error explanations
- **Network Issues**: Timeout handling and retry suggestions
- **Model Errors**: Automatic fallback messaging

## 🌍 Supported Locations

The agent supports:
- ✅ Cities (e.g., "Paris", "Tokyo", "New York")
- ✅ Regions (e.g., "Tuscany", "California")
- ✅ Countries (e.g., "Japan", "Brazil")
- ✅ Airports (e.g., "LAX", "JFK")
- ✅ Coordinates (e.g., "48.8566,2.3522")
- ✅ Postal codes (e.g., "10001", "SW1A 1AA")

## 📈 Use Cases

### Personal Applications
- Daily outfit planning
- Weekend activity decisions
- Travel packing assistance
- Event planning

### Professional Applications
- Construction project planning
- Outdoor event management
- Transportation logistics
- Agricultural planning
- Tourism recommendations

## 🐛 Troubleshooting

### Common Issues

**Issue: "GROQ_API_KEY not set"**
```bash
Solution: Set environment variable or enter in UI
export GROQ_API_KEY="your_key_here"
```

**Issue: "Weather API error: 401"**
```bash
Solution: Check your WeatherAPI key is valid
Visit: https://www.weatherapi.com/my/
```

**Issue: "Model decommissioned error"**
```bash
Solution: Update model name in get_llm() function
Use: "llama-3.3-70b-versatile"
```

**Issue: "No location specified"**
```bash
Solution: Include a city/location in your query
Example: "What should I wear in London?"
```

## 🚀 Advanced Usage

### Running with Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY weather_agent.py .

ENV GROQ_API_KEY=""
ENV WEATHER_API_KEY=""

EXPOSE 7860

CMD ["python", "weather_agent.py"]
```

Build and run:
```bash
docker build -t weather-agent .
docker run -p 7860:7860 \
  -e GROQ_API_KEY="your_key" \
  -e WEATHER_API_KEY="your_key" \
  weather-agent
```

### API Integration

You can also use the core functions programmatically:

```python
from weather_agent import create_agent_graph

# Initialize agent
agent = create_agent_graph()

# Process query
initial_state = {
    "messages": [],
    "user_query": "What should I wear in Paris?",
    "location": "",
    "weather_data": {},
    "recommendations": "",
    "error": ""
}

result = agent.invoke(initial_state)
print(result["recommendations"])
```

## 📝 Project Structure

```
weather-recommendation-agent/
│
├── weather_agent.py          # Main application file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env.example             # Example environment variables
├── LICENSE                  # MIT License
│
├── docs/                    # Additional documentation
│   ├── architecture.md      # Detailed architecture
│   └── api_reference.md     # API documentation
│
└── examples/                # Example queries and outputs
    ├── clothing_examples.md
    ├── travel_examples.md
    └── activity_examples.md
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Areas for Contribution
- Additional weather data sources
- More recommendation categories
- Multi-language support
- Mobile app version
- Enhanced error handling
- Performance optimizations


## 🙏 Acknowledgments

- **LangChain** for the excellent agent framework
- **Groq** for lightning-fast LLM inference
- **WeatherAPI.com** for reliable weather data
- **Gradio** for the intuitive UI framework

## 📧 Contact

- 📧 **Email**: harshitwaldia112@gmail.com
- 🐦 **Twitter**: [@HarshitWaldia](https://x.com/HarshitWaldia)
- 💼 **LinkedIn**: [Harshit Waldia](https://www.linkedin.com/in/harshit-waldia/)
- ⚙️ **GitHub**: [@HarshitWaldia](https://github.com/HarshitWaldia)

## 🔗 Links

- [Documentation](https://github.com/HarshitWaldia/Weather-Based-Recommendation-Agent-using-Agentic-AI/wiki)
- [Report Bug](https://github.com/HarshitWaldia/Weather-Based-Recommendation-Agent-using-Agentic-AI/issues)
- [Request Feature](https://github.com/HarshitWaldia/Weather-Based-Recommendation-Agent-using-Agentic-AI/issues)

## ⭐ Star History

If you find this project helpful, please consider giving it a star!

---
<p align="center">
  <strong>अहं ब्रह्मास्मि</strong>
</p>

*Making weather data actionable and human-centric*
