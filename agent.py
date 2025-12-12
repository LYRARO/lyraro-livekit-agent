"""
LYRARO Voice Agent - LiveKit + Deepgram + ElevenLabs + Lovable AI
Fonio-Style Architecture for German Handcraft Businesses
"""

import os
import json
import aiohttp
from datetime import datetime
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import deepgram, elevenlabs
from dotenv import load_dotenv

load_dotenv()

# Configuration
LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Edge Function URLs
EDGE_FUNCTION_BASE_URL = os.getenv("EDGE_FUNCTION_BASE_URL", "https://zistdjanhrbppnkdbanc.supabase.co/functions/v1")


async def fetch_agent_config(to_number: str, from_number: str) -> dict:
    """Fetch agent configuration from Edge Function"""
    url = f"{EDGE_FUNCTION_BASE_URL}/get-agent-config"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                url,
                json={"to_number": to_number, "from_number": from_number},
                headers={"Content-Type": "application/json"},
            ) as response:
                if response.status != 200:
                    print(f"Config fetch error: {response.status}")
                    return get_default_config()
                
                data = await response.json()
                return data.get("config", get_default_config())
        except Exception as e:
            print(f"Config fetch exception: {e}")
            return get_default_config()


def get_default_config() -> dict:
    """Default configuration if Edge Function fails"""
    return {
        "agent_id": None,
        "company_id": None,
        "company_name": "LYRARO Demo",
        "industry": "allgemeines_handwerk",
        "greeting": "Guten Tag, wie kann ich Ihnen helfen?",
        "base_prompt": "",
        "voice_id": "EXAVITQu4vr4xnSDxMaL",
        "opening_hours": "Mo-Fr 8-17 Uhr",
        "forwarding_number": "",
        "emergency_number": "",
    }


def build_system_prompt(agent_config: dict) -> str:
    """Build the complete system prompt from agent configuration"""
    company_name = agent_config.get("company_name", "Handwerksbetrieb")
    industry = agent_config.get("industry", "allgemeines_handwerk")
    greeting = agent_config.get("greeting", f"Guten Tag, {company_name}, wie kann ich Ihnen helfen?")
    base_prompt = agent_config.get("base_prompt", "")
    opening_hours = agent_config.get("opening_hours", "Mo-Fr 8-17 Uhr")
    forwarding_number = agent_config.get("forwarding_number", "")
    emergency_number = agent_config.get("emergency_number", "")
    
    industry_names = {
        "elektro": "Elektrobetrieb",
        "shk": "SHK-Betrieb (Sanitär, Heizung, Klima)",
        "tischler": "Tischlereibetrieb",
        "maler": "Malerbetrieb",
        "dachdecker": "Dachdeckerbetrieb",
        "allgemeines_handwerk": "Handwerksbetrieb",
    }
    industry_name = industry_names.get(industry, "Handwerksbetrieb")
    
    industry_contexts = {
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
    industry_context = industry_contexts.get(industry, industry_contexts["allgemeines_handwerk"])
    
    return f"""Du bist ein professioneller, freundlicher Telefonassistent für {company_name}, einen deutschen {industry_name}.

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

{industry_context}

GESCHÄFTSZEITEN: {opening_hours}
{f"WEITERLEITUNG BEI DRINGEND: {forwarding_number}" if forwarding_number else ""}
{f"NOTFALLNUMMER: {emergency_number}" if emergency_number else ""}

{base_prompt}

WICHTIG: Lies die Gesprächshistorie sorgfältig durch. NIEMALS eine Frage stellen, die bereits beantwortet wurde."""


async def send_webhook(event_type: str, payload: dict, agent_config: dict, call_id: str):
    """Send webhook to Edge Function for call logging"""
    webhook_url = f"{EDGE_FUNCTION_BASE_URL}/livekit-webhook"
    
    async with aiohttp.ClientSession() as session:
        data = {
            "event_type": event_type,
            "call_id": call_id,
            "agent_id": agent_config.get("agent_id"),
            "company_id": agent_config.get("company_id"),
            "payload": payload,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            async with session.post(webhook_url, json=data, headers={"Content-Type": "application/json"}) as response:
                if response.status != 200:
                    print(f"Webhook error: {response.status}")
                else:
                    print(f"Webhook sent: {event_type}")
        except Exception as e:
            print(f"Webhook exception: {e}")


async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the LiveKit agent"""
    
    # Extract phone numbers from room metadata
    metadata = {}
    if ctx.room.metadata:
        try:
            metadata = json.loads(ctx.room.metadata)
        except:
            pass
    
    from_number = metadata.get("from_number", "unknown")
    to_number = metadata.get("to_number", "unknown")
    call_id = ctx.room.name
    
    print(f"Call received: from={from_number}, to={to_number}")
    
    # Fetch agent configuration
    agent_config = await fetch_agent_config(to_number, from_number)
    print(f"Loaded config for: {agent_config.get('company_name')}")
    
    # Send call started webhook
    await send_webhook("call_started", {
        "from_number": from_number,
        "to_number": to_number,
    }, agent_config, call_id)
    
    # Connect to the room
    await ctx.connect()
    
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
        voice_id=agent_config.get("voice_id", "EXAVITQu4vr4xnSDxMaL"),
        model="eleven_turbo_v2_5",
    )
    
    # Create agent session
    session = AgentSession(
        stt=stt,
        tts=tts,
    )
    
    # Build system prompt
    system_prompt = build_system_prompt(agent_config)
    
    # Start the session with the agent
    await session.start(
        room=ctx.room,
        agent=Agent(instructions=system_prompt),
    )
    
    # Send initial greeting
    greeting = agent_config.get("greeting", "Guten Tag, wie kann ich Ihnen helfen?")
    await session.generate_reply(instructions=f"Sage exakt: {greeting}")
    
    # Wait for session to end
    await session.wait_for_close()
    
    # Send call ended webhook
    await send_webhook("call_ended", {
        "duration_seconds": (datetime.now() - datetime.now()).total_seconds(),
    }, agent_config, call_id)


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET,
            ws_url=LIVEKIT_URL,
        )
    )

