"""
Weather-Based Recommendation Agent using Agentic AI
Built with LangGraph, Groq, and Gradio
"""

import os
import json
import requests
from typing import TypedDict, Annotated, Literal
from datetime import datetime
import gradio as gr

# LangChain and LangGraph imports
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


# CONFIGURATION


# API Keys - Set these as environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")


# STATE DEFINITION


class AgentState(TypedDict):
    """State object that flows through the agent workflow"""

    messages: Annotated[list, add_messages]
    user_query: str
    location: str
    weather_data: dict
    recommendations: str
    error: str


# LLM INITIALIZATION


def get_llm():
    """Initialize Groq LLM with LLaMA 3.1"""
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY not set. Please set it as an environment variable."
        )

    return ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
    )


# WEATHER API FUNCTIONS


def fetch_weather_data(location: str) -> dict:
    """
    Fetch real-time and forecast weather data from WeatherAPI.com
    """
    if not WEATHER_API_KEY:
        return {"error": "WEATHER_API_KEY not set"}

    try:
        # Fetch current weather and 3-day forecast
        url = f"http://api.weatherapi.com/v1/forecast.json"
        params = {
            "key": WEATHER_API_KEY,
            "q": location,
            "days": 3,
            "aqi": "no",
            "alerts": "yes",
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()

        # Extract relevant weather information
        current = data.get("current", {})
        forecast_days = data.get("forecast", {}).get("forecastday", [])
        location_info = data.get("location", {})

        weather_summary = {
            "location": f"{location_info.get('name')}, {location_info.get('country')}",
            "local_time": location_info.get("localtime"),
            "current": {
                "temp_c": current.get("temp_c"),
                "temp_f": current.get("temp_f"),
                "condition": current.get("condition", {}).get("text"),
                "wind_kph": current.get("wind_kph"),
                "wind_mph": current.get("wind_mph"),
                "humidity": current.get("humidity"),
                "feelslike_c": current.get("feelslike_c"),
                "feelslike_f": current.get("feelslike_f"),
                "uv": current.get("uv"),
                "visibility_km": current.get("vis_km"),
            },
            "forecast": [],
        }

        # Process forecast data
        for day in forecast_days:
            day_data = day.get("day", {})
            weather_summary["forecast"].append(
                {
                    "date": day.get("date"),
                    "max_temp_c": day_data.get("maxtemp_c"),
                    "min_temp_c": day_data.get("mintemp_c"),
                    "max_temp_f": day_data.get("maxtemp_f"),
                    "min_temp_f": day_data.get("mintemp_f"),
                    "condition": day_data.get("condition", {}).get("text"),
                    "rain_chance": day_data.get("daily_chance_of_rain"),
                    "snow_chance": day_data.get("daily_chance_of_snow"),
                    "max_wind_kph": day_data.get("maxwind_kph"),
                    "avg_humidity": day_data.get("avghumidity"),
                    "uv": day_data.get("uv"),
                }
            )

        # Add alerts if any
        alerts = data.get("alerts", {}).get("alert", [])
        if alerts:
            weather_summary["alerts"] = [
                {
                    "headline": alert.get("headline"),
                    "severity": alert.get("severity"),
                    "event": alert.get("event"),
                }
                for alert in alerts
            ]

        return weather_summary

    except requests.RequestException as e:
        return {"error": f"Weather API error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# AGENT NODES


def extract_location_node(state: AgentState) -> AgentState:
    """
    Node 1: Extract location from user query using LLM
    """
    try:
        llm = get_llm()
        user_query = state["user_query"]

        system_prompt = """You are a location extraction expert. Extract the primary geographic location from the user's query.
        
Rules:
- Return ONLY the location name (city, region, or country)
- If multiple locations are mentioned, return the primary one
- If no location is mentioned, return "unknown"
- Do not include any explanation, just the location name

Examples:
User: "What should I wear in Paris tomorrow?"
Response: Paris

User: "Is it safe to travel to Tokyo next week?"
Response: Tokyo

User: "What's the weather like?"
Response: unknown"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_query),
        ]

        response = llm.invoke(messages)
        location = response.content.strip()

        state["location"] = location
        state["messages"].append(AIMessage(content=f"Extracted location: {location}"))

        return state

    except Exception as e:
        state["error"] = f"Location extraction error: {str(e)}"
        return state


def fetch_weather_node(state: AgentState) -> AgentState:
    """
    Node 2: Fetch weather data from API
    """
    location = state.get("location", "unknown")

    if location == "unknown" or not location:
        state["error"] = (
            "No location specified. Please include a city or location in your query."
        )
        return state

    try:
        weather_data = fetch_weather_data(location)

        if "error" in weather_data:
            state["error"] = weather_data["error"]
        else:
            state["weather_data"] = weather_data
            state["messages"].append(
                AIMessage(
                    content=f"Fetched weather data for {weather_data.get('location')}"
                )
            )

        return state

    except Exception as e:
        state["error"] = f"Weather fetch error: {str(e)}"
        return state


def analyze_and_recommend_node(state: AgentState) -> AgentState:
    """
    Node 3: Analyze weather data and generate recommendations
    """
    try:
        llm = get_llm()
        user_query = state["user_query"]
        weather_data = state.get("weather_data", {})

        if not weather_data:
            state["error"] = "No weather data available"
            return state

        # Format weather data for LLM
        weather_context = json.dumps(weather_data, indent=2)

        system_prompt = """You are a weather-based recommendation expert. Analyze the provided weather data and user query to generate practical, actionable recommendations.

Your recommendations should cover:
1. **Clothing Suggestions**: Based on temperature, wind, rain, and UV index
2. **Travel Safety**: Assess any weather-related risks
3. **Activity Planning**: Suggest best times and precautions
4. **Health Considerations**: UV protection, hydration, etc.

Guidelines:
- Be specific and practical
- Consider current conditions AND forecast
- Mention any weather alerts if present
- Use natural, conversational language
- Focus on what matters most to the user's intent
- Include temperature in both Celsius and Fahrenheit when relevant

Format your response in clear sections with appropriate headers."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"""User Query: {user_query}

Weather Data:
{weather_context}

Provide comprehensive recommendations based on this information."""
            ),
        ]

        response = llm.invoke(messages)
        recommendations = response.content

        state["recommendations"] = recommendations
        state["messages"].append(AIMessage(content="Generated recommendations"))

        return state

    except Exception as e:
        state["error"] = f"Recommendation generation error: {str(e)}"
        return state


def error_handler_node(state: AgentState) -> AgentState:
    """
    Handle errors gracefully
    """
    error_msg = state.get("error", "Unknown error occurred")
    state["recommendations"] = (
        f"⚠️ **Error**: {error_msg}\n\nPlease try again with a different query or check your API keys."
    )
    return state


# ROUTING LOGIC


def should_continue(state: AgentState) -> Literal["fetch_weather", "error"]:
    """Route after location extraction"""
    if state.get("error"):
        return "error"
    return "fetch_weather"


def after_weather_fetch(state: AgentState) -> Literal["analyze", "error"]:
    """Route after weather fetch"""
    if state.get("error"):
        return "error"
    return "analyze"


# GRAPH CONSTRUCTION


def create_agent_graph():
    """
    Build the LangGraph workflow
    """
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("extract_location", extract_location_node)
    workflow.add_node("fetch_weather", fetch_weather_node)
    workflow.add_node("analyze", analyze_and_recommend_node)
    workflow.add_node("error", error_handler_node)

    # Define edges
    workflow.set_entry_point("extract_location")

    workflow.add_conditional_edges(
        "extract_location",
        should_continue,
        {"fetch_weather": "fetch_weather", "error": "error"},
    )

    workflow.add_conditional_edges(
        "fetch_weather", after_weather_fetch, {"analyze": "analyze", "error": "error"}
    )

    workflow.add_edge("analyze", END)
    workflow.add_edge("error", END)

    return workflow.compile()


# GRADIO INTERFACE


def process_query(user_query: str, groq_key: str, weather_key: str) -> str:
    """
    Process user query through the agent workflow
    """
    # Set API keys
    global GROQ_API_KEY, WEATHER_API_KEY
    GROQ_API_KEY = groq_key or GROQ_API_KEY
    WEATHER_API_KEY = weather_key or WEATHER_API_KEY

    if not GROQ_API_KEY:
        return "❌ Please provide your Groq API key"
    if not WEATHER_API_KEY:
        return "❌ Please provide your WeatherAPI key"

    if not user_query.strip():
        return "❌ Please enter a query"

    try:
        # Initialize state
        initial_state = {
            "messages": [],
            "user_query": user_query,
            "location": "",
            "weather_data": {},
            "recommendations": "",
            "error": "",
        }

        # Create and run agent
        agent = create_agent_graph()
        result = agent.invoke(initial_state)

        # Return recommendations
        return result.get("recommendations", "No recommendations generated")

    except Exception as e:
        return f"❌ **Error**: {str(e)}\n\nPlease check your API keys and try again."


# Example queries for users
EXAMPLES = [
    "What should I wear in London tomorrow?",
    "Is it safe to travel to Mumbai next week?",
    "What clothes should I pack for a trip to Tokyo?",
    "Will the weather affect my outdoor plans in New York this weekend?",
    "Should I bring an umbrella to Paris?",
    "What's the best time to visit Sydney for beach activities?",
]


def create_gradio_interface():
    """
    Create Gradio UI
    """
    with gr.Blocks(
        title="Weather-Based Recommendation Agent", theme=gr.themes.Soft()
    ) as demo:
        gr.Markdown(
            """
        # 🌤️ Weather-Based Recommendation Agent
        
        Get intelligent, context-aware recommendations based on real-time weather data.
        
        **Ask questions like:**
        - "What should I wear in Paris tomorrow?"
        - "Is it safe to travel to Tokyo next week?"
        - "What clothes should I pack for London?"
        """
        )

        with gr.Row():
            with gr.Column(scale=1):
                groq_key_input = gr.Textbox(
                    label="🔑 Groq API Key",
                    placeholder="Enter your Groq API key (gsk_...)",
                    type="password",
                    value=GROQ_API_KEY,
                )

                weather_key_input = gr.Textbox(
                    label="🔑 WeatherAPI Key",
                    placeholder="Enter your WeatherAPI.com key",
                    type="password",
                    value=WEATHER_API_KEY,
                )

                gr.Markdown(
                    """
                **Get your API keys:**
                - [Groq API Key](https://console.groq.com/)
                - [WeatherAPI Key](https://www.weatherapi.com/) (Free tier available)
                """
                )

        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(
                    label="💬 Your Weather Query",
                    placeholder="e.g., What should I wear in Paris tomorrow?",
                    lines=3,
                )

                submit_btn = gr.Button(
                    "🚀 Get Recommendations", variant="primary", size="lg"
                )

                gr.Examples(
                    examples=EXAMPLES, inputs=query_input, label="Example Queries"
                )

        with gr.Row():
            output = gr.Markdown(label="📋 Recommendations")

        submit_btn.click(
            fn=process_query,
            inputs=[query_input, groq_key_input, weather_key_input],
            outputs=output,
        )

        gr.Markdown(
            """
        ---
        ### 📚 About This Agent
        
        This intelligent agent uses:
        - **LangGraph** for orchestrated agentic workflow
        - **Groq + LLaMA 3.1** for fast, efficient reasoning
        - **WeatherAPI.com** for real-time weather data
        
        The agent follows a structured pipeline:
        1. **Extract Location** - Identifies the location from your query
        2. **Fetch Weather** - Gets current and forecast data
        3. **Analyze & Recommend** - Generates personalized recommendations
        
        **Features:**
        - Clothing suggestions based on temperature, wind, and rain
        - Travel safety assessments
        - Activity planning guidance
        - Health considerations (UV, humidity)
        - Weather alert notifications
        """
        )

    return demo


# MAIN EXECUTION

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch()
