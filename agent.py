"""
LYRARO Voice Agent - LiveKit + Deepgram + ElevenLabs + Lovable AI
Fonio-Style Architecture for German Handcraft Businesses
"""

import os
import json
import asyncio
import aiohttp
from datetime import datetime
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import deepgram, elevenlabs
from dotenv import load_dotenv

load_dotenv()

# Configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
LOVABLE_API_KEY = os.getenv("LOVABLE_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Lovable AI Gateway
LOVABLE_AI_URL = "https://ai.gateway.lovable.dev/v1/chat/completions"


class LyraroVoiceAgent:
    """Voice agent for handling inbound calls"""
    
    def __init__(self, agent_config: dict):
        self.agent_config = agent_config
        self.conversation_history: list[dict] = []
        self.collected_data = {
            "customer_name": None,
            "customer_phone": None,
            "customer_address": None,
            "problem_type": None,
            "work_description": None,
            "urgency": None,
            "requested_date": None,
            "requested_time": None,
            "callback_requested": False,
        }
        self.call_start_time = datetime.now()
        self.call_id = None
        
    def build_system_prompt(self) -> str:
        """Build the complete system prompt from agent configuration"""
        company_name = self.agent_config.get("company_name", "Handwerksbetrieb")
        industry = self.agent_config.get("industry", "allgemeines_handwerk")
        greeting = self.agent_config.get("greeting", f"Guten Tag, {company_name}, wie kann ich Ihnen helfen?")
        base_prompt = self.agent_config.get("base_prompt", "")
        opening_hours = self.agent_config.get("opening_hours", "Mo-Fr 8-17 Uhr")
        forwarding_number = self.agent_config.get("forwarding_number", "")
        emergency_number = self.agent_config.get("emergency_number", "")
        
        system_prompt = f"""Du bist ein professioneller, freundlicher Telefonassistent für {company_name}, einen deutschen {self._get_industry_name(industry)}.

DEINE ERSTE AUSSAGE MUSS EXAKT SEIN: "{greeting}"
Nach der Begrüßung begrüße NIEMALS ein zweites Mal.

GESPRÄCHSABLAUF:
1. Begrüßungsphase: Nur die konfigurierte Begrüßung, dann WARTEN auf Anrufer
2. Anliegen erfassen: Aktiv zuhören, kurz zusammenfassen, EINE Rückfrage stellen (maximal 1-2 pro Runde)
3. Vertiefende Rückfragen: Kontextabhängig nach Details fragen (Gerätetyp, Standort, Dringlichkeit)
4. Datenerfassung: NACH Verständnis des Problems einzeln fragen: Name, Rückrufnummer, Adresse, Terminwunsch
5. Abschluss: Kurze Zusammenfassung, dann "Vielen Dank für Ihren Anruf. Der Betrieb meldet sich bei Ihnen. Auf Wiederhören."

KOMMUNIKATIONSREGELN:
- Kurze, natürliche Sätze (maximal 1-2 Sätze pro Antwort)
- NIEMALS Informationen erfragen, die bereits genannt wurden
- NIEMALS mehrere Fragen gleichzeitig stellen
- Bei unklaren Namen: "Könnten Sie Ihren Namen bitte buchstabieren?" und "Ist das mit Umlaut?"
- Anrufer ausreden lassen, WARTEN vor Antwort
- Bei Stille: Geduldig warten, erst nach 15+ Sekunden fragen "Hallo, sind Sie noch da?"

TERMINVEREINBARUNG:
- Du kannst KEINE verbindlichen Termine zusagen
- Nur Terminwünsche notieren
- Sag: "Ich notiere Ihren Wunschtermin. Ein Mitarbeiter meldet sich, um das Genauere zu klären."

GRENZEN:
- Keine technischen Diagnosen stellen
- Keine Preisauskünfte geben
- Keine festen Zusagen machen
- Bei Notfall (Feuer, Gas, Stromschlag): Sofort auf 112 verweisen

{self._get_industry_context(industry)}

GESCHÄFTSZEITEN: {opening_hours}
{f"WEITERLEITUNG BEI DRINGEND: {forwarding_number}" if forwarding_number else ""}
{f"NOTFALLNUMMER: {emergency_number}" if emergency_number else ""}

{base_prompt}

WICHTIG: Lies die Gesprächshistorie sorgfältig durch. NIEMALS eine Frage stellen, die bereits beantwortet wurde."""

        return system_prompt
    
    def _get_industry_name(self, industry: str) -> str:
        """Get human-readable industry name"""
        names = {
            "elektro": "Elektrobetrieb",
            "shk": "SHK-Betrieb (Sanitär, Heizung, Klima)",
            "tischler": "Tischlereibetrieb",
            "maler": "Malerbetrieb",
            "dachdecker": "Dachdeckerbetrieb",
            "allgemeines_handwerk": "Handwerksbetrieb",
        }
        return names.get(industry, "Handwerksbetrieb")
    
    def _get_industry_context(self, industry: str) -> str:
        """Get industry-specific context for the prompt"""
        contexts = {
            "elektro": """BRANCHENSPEZIFISCH (Elektro):
- Frage nach Art des Problems: Stromausfall, Kurzschluss, defekte Steckdose, Sicherung
- Frage nach Anlagetyp: Hausinstallation, Smart Home, Wallbox, Photovoltaik
- Bei Stromausfall: Prüfen ob Sicherung rausgeflogen, ob Nachbarn auch betroffen
- SICHERHEIT: Bei Brandgeruch oder Funken → sofort Strom abstellen, ggf. 112""",
            
            "shk": """BRANCHENSPEZIFISCH (Sanitär/Heizung/Klima):
- Frage nach Art des Problems: Heizungsausfall, Wasserrohrbruch, verstopfter Abfluss, Warmwasser
- Frage nach Anlagetyp: Gas, Öl, Wärmepumpe, Solar
- Bei Heizungsausfall im Winter: Dringlichkeit erfassen ("Sitzen Sie im Kalten?")
- Bei Wasseraustritt: Haupthahn abdrehen lassen, Dringlichkeit hoch""",
            
            "tischler": """BRANCHENSPEZIFISCH (Tischlerei):
- Frage nach Art des Auftrags: Möbelbau, Reparatur, Einbauschrank, Fenster/Türen
- Frage nach Maßen falls relevant
- Frage nach Materialwunsch: Massivholz, Furnier, Holzart
- Zeitrahmen: Neubau oder Reparatur?""",
            
            "maler": """BRANCHENSPEZIFISCH (Maler):
- Frage nach Art der Arbeit: Innenanstrich, Außenanstrich, Tapezieren, Fassade
- Frage nach Fläche/Raumgröße
- Frage nach Zustand: Risse, Abblätterungen, Feuchtigkeit
- Farbwünsche notieren falls genannt""",
            
            "dachdecker": """BRANCHENSPEZIFISCH (Dachdecker):
- Frage nach Art des Problems: Undichtigkeit, Sturmschaden, Dachrinne, Dämmung
- Frage nach Dachtyp: Satteldach, Flachdach, Schindeln, Ziegel
- Bei Wassereintritt: Dringlichkeit hoch, provisorische Abdeckung erwähnen
- Sicherheitsrelevant: Keine Eigenarbeiten auf dem Dach empfehlen""",
            
            "allgemeines_handwerk": """BRANCHENSPEZIFISCH (Allgemein):
- Frage nach Art des Handwerks/der Dienstleistung
- Erfasse Details zum Umfang der Arbeit
- Frage nach Zeitrahmen und Dringlichkeit""",
        }
        return contexts.get(industry, contexts["allgemeines_handwerk"])

    async def call_lovable_ai(self, messages: list[dict]) -> str:
        """Call Lovable AI Gateway for LLM response"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": "google/gemini-2.5-flash",
                "messages": messages,
                "max_tokens": 250,
                "temperature": 0.6,
            }
            
            headers = {
                "Authorization": f"Bearer {LOVABLE_API_KEY}",
                "Content-Type": "application/json",
            }
            
            async with session.post(LOVABLE_AI_URL, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"Lovable AI error: {response.status} - {error_text}")
                    return "Entschuldigung, es gab ein technisches Problem. Können Sie das bitte wiederholen?"
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]

    async def process_user_input(self, text: str) -> str:
        """Process user input and generate response"""
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": text,
            "timestamp": datetime.now().isoformat()
        })
        
        # Build messages for LLM
        messages = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        
        # Add conversation history
        for msg in self.conversation_history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Get LLM response
        response = await self.call_lovable_ai(messages)
        
        # Add assistant response to history
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat()
        })
        
        return response

    async def send_webhook(self, event_type: str, payload: dict):
        """Send webhook to Supabase for call logging"""
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            print("Supabase not configured, skipping webhook")
            return
            
        webhook_url = f"{SUPABASE_URL}/functions/v1/livekit-webhook"
        
        async with aiohttp.ClientSession() as session:
            data = {
                "event_type": event_type,
                "call_id": self.call_id,
                "agent_id": self.agent_config.get("agent_id"),
                "company_id": self.agent_config.get("company_id"),
                "payload": payload,
                "timestamp": datetime.now().isoformat()
            }
            
            headers = {
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            }
            
            try:
                async with session.post(webhook_url, json=data, headers=headers) as response:
                    if response.status != 200:
                        print(f"Webhook error: {response.status}")
            except Exception as e:
                print(f"Webhook exception: {e}")

    def get_transcript(self) -> str:
        """Get formatted transcript of the conversation"""
        lines = []
        for msg in self.conversation_history:
            role = "Anrufer" if msg["role"] == "user" else "Assistent"
            lines.append(f"{role}: {msg['content']}")
        return "\n".join(lines)


async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the LiveKit agent"""
    
    # Get agent configuration from room metadata or SIP headers
    room = ctx.room
    metadata = json.loads(room.metadata) if room.metadata else {}
    
    # Extract agent config from metadata (set by SIP dispatch rules)
    agent_config = metadata.get("agent_config", {
        "company_name": "LYRARO Demo",
        "industry": "allgemeines_handwerk",
        "greeting": "Guten Tag, LYRARO Demo, wie kann ich Ihnen helfen?",
        "agent_id": None,
        "company_id": None,
    })
    
    # Initialize the voice agent
    voice_agent = LyraroVoiceAgent(agent_config)
    voice_agent.call_id = room.name  # Use room name as call ID
    
    # Send call started webhook
    await voice_agent.send_webhook("call_started", {
        "from_number": metadata.get("from_number", "unknown"),
        "to_number": metadata.get("to_number", "unknown"),
    })
    
    # Configure STT (Deepgram)
    stt = deepgram.STT(
        api_key=DEEPGRAM_API_KEY,
        model="nova-2",
        language="de",
        punctuate=True,
        interim_results=True,
    )
    
    # Configure TTS (ElevenLabs)
    tts = elevenlabs.TTS(
        api_key=ELEVENLABS_API_KEY,
        voice_id=agent_config.get("voice_id", "EXAVITQu4vr4xnSDxMaL"),  # Sarah
        model="eleven_turbo_v2_5",
        language_code="de",
    )
    
    # Create the agent session
    session = AgentSession(
        stt=stt,
        tts=tts,
        chat_ctx=ChatContext(),
    )
    
    # Track if we've sent the greeting
    greeting_sent = False
    
    @session.on("user_speech_committed")
    async def on_user_speech(text: str):
        """Handle transcribed user speech"""
        nonlocal greeting_sent
        
        # Filter out non-German hallucinations
        if not is_mostly_latin(text):
            print(f"Filtered non-Latin transcript: {text[:50]}...")
            return
            
        print(f"User said: {text}")
        
        # Process and respond
        response = await voice_agent.process_user_input(text)
        print(f"Agent responds: {response}")
        
        await session.say(response)
    
    @session.on("agent_started_speaking")
    async def on_agent_started():
        print("Agent started speaking")
    
    @session.on("agent_stopped_speaking") 
    async def on_agent_stopped():
        print("Agent stopped speaking")
    
    # Start the session
    await session.start(
        room=room,
        participant=None,  # Will connect to first participant
        room_input_options=RoomInputOptions(
            noise_cancellation=True,
        ),
    )
    
    # Send initial greeting
    greeting = agent_config.get("greeting", "Guten Tag, wie kann ich Ihnen helfen?")
    voice_agent.conversation_history.append({
        "role": "assistant",
        "content": greeting,
        "timestamp": datetime.now().isoformat()
    })
    await session.say(greeting)
    greeting_sent = True
    
    # Wait for the session to end
    await session.wait_for_close()
    
    # Send call ended webhook with transcript
    await voice_agent.send_webhook("call_ended", {
        "transcript": voice_agent.get_transcript(),
        "conversation_history": voice_agent.conversation_history,
        "collected_data": voice_agent.collected_data,
        "duration_seconds": (datetime.now() - voice_agent.call_start_time).total_seconds(),
    })


def is_mostly_latin(text: str) -> bool:
    """Check if text is mostly Latin characters (filter Whisper hallucinations)"""
    if not text:
        return False
    latin_count = sum(1 for c in text if c.isascii() or c in 'äöüÄÖÜß')
    return latin_count / len(text) > 0.5


if __name__ == "__main__":
    # Run the agent worker
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            ws_url=LIVEKIT_URL,
        )
    )
