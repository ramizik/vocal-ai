"""
Enhanced Letta Service with Progressive Vocal Personality Development and Health Monitoring
Implements multi-agent architecture with specialized memory blocks for vocal coaching
"""
import os
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import uuid
from dataclasses import dataclass, asdict
from enum import Enum
import pytz
import numpy as np
from supabase import create_client, Client

logger = logging.getLogger(__name__)

class VocalPersonalityType(Enum):
    ANALYTICAL = "analytical"
    EXPRESSIVE = "expressive"
    METHODICAL = "methodical"
    INTUITIVE = "intuitive"
    PERFECTIONIST = "perfectionist"

class HealthRiskLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class VocalPersonalityProfile:
    """Dynamic vocal personality profile that evolves over time"""
    user_id: str
    personality_type: VocalPersonalityType
    evolution_score: float  # 0-10 scale
    breakthrough_moments: List[Dict[str, Any]]
    learning_preferences: Dict[str, Any]
    coaching_adaptations: Dict[str, Any]
    vocal_fingerprint: Dict[str, Any]
    last_evolution_date: datetime
    total_evolution_points: int
    created_at: datetime
    updated_at: datetime

@dataclass
class VocalHealthProfile:
    """Comprehensive vocal health monitoring profile"""
    user_id: str
    current_risk_level: HealthRiskLevel
    strain_patterns: Dict[str, Any]
    recovery_patterns: Dict[str, Any]
    environmental_correlations: Dict[str, Any]
    warning_indicators: List[Dict[str, Any]]
    optimal_practice_windows: List[Dict[str, Any]]
    health_trends: Dict[str, Any]
    last_health_check: datetime
    created_at: datetime
    updated_at: datetime

class EnhancedLettaService:
    """Enhanced Letta service with multi-agent architecture"""

    def __init__(self):
        # Initialize Supabase client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        if not supabase_url or not supabase_key:
            logger.warning("Supabase credentials not found. Using mock data.")
            self.supabase: Optional[Client] = None
        else:
            try:
                self.supabase = create_client(supabase_url, supabase_key)
                logger.info("Enhanced Letta Supabase client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {str(e)}")
                self.supabase = None

        # Initialize Letta client with multiple agents
        self.letta_api_key = os.getenv("LETTA_API_KEY")
        self.agents = {
            "personality_coach": os.getenv("LETTA_PERSONALITY_AGENT_ID"),
            "health_monitor": os.getenv("LETTA_HEALTH_AGENT_ID"),
            "pattern_analyst": os.getenv("LETTA_PATTERN_AGENT_ID")
        }

        if not self.letta_api_key:
            logger.warning("Letta API key not found. Using mock responses.")
            self.use_ai = False
            self.letta_client = None
        else:
            try:
                from letta_client import Letta
                self.letta_client = Letta(token=self.letta_api_key)
                self.use_ai = True
                logger.info("Enhanced Letta AI client initialized with multi-agent support")

                # Create agents if they don't exist
                asyncio.create_task(self._initialize_agents())

            except ImportError as e:
                logger.error(f"Failed to import Letta client: {str(e)}")
                self.use_ai = False
                self.letta_client = None

    async def _initialize_agents(self):
        """Initialize specialized Letta agents for different functions"""
        if not self.letta_client:
            return

        agents_config = [
            {
                "key": "personality_coach",
                "name": "Personality Coach",
                "memory_blocks": [
                    {
                        "label": "vocal_personality_core",
                        "description": "Core vocal characteristics and evolving personality traits based on user's practice patterns, preferences, and breakthrough moments",
                        "value": "",
                        "limit": 8000
                    },
                    {
                        "label": "breakthrough_moments",
                        "description": "Significant vocal breakthroughs, learning milestones, and memorable coaching moments that shape personality development",
                        "value": "",
                        "limit": 6000
                    },
                    {
                        "label": "coaching_adaptation",
                        "description": "Adaptive coaching style preferences and effective methods based on personality type and learning progress",
                        "value": "",
                        "limit": 5000
                    }
                ]
            },
            {
                "key": "health_monitor",
                "name": "Health Monitor",
                "memory_blocks": [
                    {
                        "label": "vocal_strain_patterns",
                        "description": "Tracks vocal stress indicators, warning signs, and patterns that lead to vocal fatigue or strain",
                        "value": "",
                        "limit": 7000
                    },
                    {
                        "label": "recovery_optimization",
                        "description": "Learning optimal rest periods, recovery strategies, and rehabilitation techniques based on individual patterns",
                        "value": "",
                        "limit": 5000
                    },
                    {
                        "label": "environmental_correlations",
                        "description": "Weather, stress, seasonal, and environmental factor impacts on vocal performance and health",
                        "value": "",
                        "limit": 6000
                    }
                ]
            },
            {
                "key": "pattern_analyst",
                "name": "Pattern Analyst",
                "memory_blocks": [
                    {
                        "label": "long_term_trends",
                        "description": "Long-term vocal development patterns, trends, and correlations across months and years of practice",
                        "value": "",
                        "limit": 8000
                    },
                    {
                        "label": "behavioral_insights",
                        "description": "User behavior patterns, practice habits, and correlations with vocal improvement",
                        "value": "",
                        "limit": 6000
                    }
                ]
            }
        ]

        try:
            for config in agents_config:
                self._get_or_create_agent(
                    agent_key=config["key"],
                    friendly_name=config["name"],
                    memory_blocks=config["memory_blocks"]
                )
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")

    def _get_or_create_agent(
        self,
        agent_key: str,
        friendly_name: str,
        memory_blocks: List[Dict[str, Any]]
    ) -> Optional[str]:
        """Helper to retrieve or create a Letta agent if it doesn't exist"""
        if self.agents.get(agent_key):
            return self.agents[agent_key]

        try:
            agent = self.letta_client.agents.create(
                memory_blocks=memory_blocks,
                tools=["web_search", "run_code"],
                model="openai/gpt-4o-mini",
                embedding="openai/text-embedding-3-small"
            )
            self.agents[agent_key] = agent.id
            logger.info(f"Created {friendly_name} Agent: {agent.id}")
            return agent.id
        except Exception as e:
            logger.error(f"Failed to create {friendly_name} Agent: {str(e)}")
            return None

    async def get_personality_profile(self, user_id: str) -> VocalPersonalityProfile:
        """Get or create user's vocal personality profile"""
        if not self.supabase:
            return self._get_fallback_personality_profile(user_id)

        try:
            response = self.supabase.table('vocal_personality_profiles').select('*').eq(
                'user_id', user_id
            ).execute()

            if response.data:
                profile_data = response.data[0]
                return VocalPersonalityProfile(
                    user_id=profile_data['user_id'],
                    personality_type=VocalPersonalityType(profile_data['personality_type']),
                    evolution_score=profile_data['evolution_score'],
                    breakthrough_moments=profile_data['breakthrough_moments'],
                    learning_preferences=profile_data['learning_preferences'],
                    coaching_adaptations=profile_data['coaching_adaptations'],
                    vocal_fingerprint=profile_data['vocal_fingerprint'],
                    last_evolution_date=self._parse_datetime_with_tz(profile_data['last_evolution_date']),
                    total_evolution_points=profile_data['total_evolution_points'],
                    created_at=self._parse_datetime_with_tz(profile_data['created_at']),
                    updated_at=self._parse_datetime_with_tz(profile_data['updated_at'])
                )
            else:
                # Create new personality profile
                new_profile = VocalPersonalityProfile(
                    user_id=user_id,
                    personality_type=VocalPersonalityType.ANALYTICAL,  # Default starting type
                    evolution_score=1.0,
                    breakthrough_moments=[],
                    learning_preferences={"style": "visual", "pace": "moderate"},
                    coaching_adaptations={"tone": "encouraging", "detail_level": "medium"},
                    vocal_fingerprint={},
                    last_evolution_date=datetime.now(pytz.utc),
                    total_evolution_points=0,
                    created_at=datetime.now(pytz.utc),
                    updated_at=datetime.now(pytz.utc)
                )

                # Save to database
                profile_dict = asdict(new_profile)
                profile_dict['personality_type'] = profile_dict['personality_type'].value
                profile_dict['last_evolution_date'] = profile_dict['last_evolution_date'].isoformat()
                profile_dict['created_at'] = profile_dict['created_at'].isoformat()
                profile_dict['updated_at'] = profile_dict['updated_at'].isoformat()

                self.supabase.table('vocal_personality_profiles').insert(profile_dict).execute()
                return new_profile

        except Exception as e:
            logger.error(f"Error getting personality profile: {str(e)}")
            return self._get_fallback_personality_profile(user_id)

    async def get_health_profile(self, user_id: str) -> VocalHealthProfile:
        """Get or create user's vocal health profile"""
        if not self.supabase:
            return self._get_fallback_health_profile(user_id)

        try:
            response = self.supabase.table('vocal_health_profiles').select('*').eq(
                'user_id', user_id
            ).execute()

            if response.data:
                health_data = response.data[0]
                return VocalHealthProfile(
                    user_id=health_data['user_id'],
                    current_risk_level=HealthRiskLevel(health_data['current_risk_level']),
                    strain_patterns=health_data['strain_patterns'],
                    recovery_patterns=health_data['recovery_patterns'],
                    environmental_correlations=health_data['environmental_correlations'],
                    warning_indicators=health_data['warning_indicators'],
                    optimal_practice_windows=health_data['optimal_practice_windows'],
                    health_trends=health_data['health_trends'],
                    last_health_check=self._parse_datetime_with_tz(health_data['last_health_check']),
                    created_at=self._parse_datetime_with_tz(health_data['created_at']),
                    updated_at=self._parse_datetime_with_tz(health_data['updated_at'])
                )
            else:
                # Create new health profile
                new_profile = VocalHealthProfile(
                    user_id=user_id,
                    current_risk_level=HealthRiskLevel.LOW,
                    strain_patterns={},
                    recovery_patterns={},
                    environmental_correlations={},
                    warning_indicators=[],
                    optimal_practice_windows=[],
                    health_trends={},
                    last_health_check=datetime.now(pytz.utc),
                    created_at=datetime.now(pytz.utc),
                    updated_at=datetime.now(pytz.utc)
                )

                # Save to database
                health_dict = asdict(new_profile)
                health_dict['current_risk_level'] = health_dict['current_risk_level'].value
                health_dict['last_health_check'] = health_dict['last_health_check'].isoformat()
                health_dict['created_at'] = health_dict['created_at'].isoformat()
                health_dict['updated_at'] = health_dict['updated_at'].isoformat()

                self.supabase.table('vocal_health_profiles').insert(health_dict).execute()
                return new_profile

        except Exception as e:
            logger.error(f"Error getting health profile: {str(e)}")
            return self._get_fallback_health_profile(user_id)

    async def analyze_vocal_evolution(self, user_id: str, vocal_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze vocal evolution and update personality development"""
        try:
            personality_profile = await self.get_personality_profile(user_id)

            # Calculate evolution metrics
            evolution_data = {
                "consistency_improvement": self._calculate_consistency_improvement(vocal_metrics),
                "technique_advancement": self._calculate_technique_advancement(vocal_metrics),
                "breakthrough_detection": self._detect_breakthrough_moments(vocal_metrics, personality_profile),
                "personality_adaptation": self._adapt_personality_type(vocal_metrics, personality_profile)
            }

            # Update personality profile if significant changes detected
            if evolution_data["breakthrough_detection"]["has_breakthrough"]:
                await self._update_personality_profile(user_id, evolution_data)

            return evolution_data

        except Exception as e:
            logger.error(f"Error analyzing vocal evolution: {str(e)}")
            return {"error": str(e)}

    async def monitor_vocal_health(self, user_id: str, vocal_metrics: Dict[str, Any], environmental_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Monitor vocal health and provide early warning system"""
        try:
            health_profile = await self.get_health_profile(user_id)

            # Health analysis
            health_analysis = {
                "strain_indicators": self._analyze_strain_indicators(vocal_metrics),
                "recovery_needs": self._assess_recovery_needs(vocal_metrics, health_profile),
                "environmental_impact": self._assess_environmental_impact(environmental_data, health_profile),
                "risk_level": self._calculate_risk_level(vocal_metrics, health_profile),
                "recommendations": self._generate_health_recommendations(vocal_metrics, health_profile)
            }

            # Update health profile
            await self._update_health_profile(user_id, health_analysis)

            return health_analysis

        except Exception as e:
            logger.error(f"Error monitoring vocal health: {str(e)}")
            return {"error": str(e)}

    async def get_contextual_coaching(self, user_id: str, context: str, user_message: str) -> Dict[str, Any]:
        """Get contextual coaching based on personality and health profiles"""
        try:
            personality_profile = await self.get_personality_profile(user_id)
            health_profile = await self.get_health_profile(user_id)

            # Determine which agent to use based on context
            agent_id = self._select_agent_for_context(context)

            if not agent_id or not self.letta_client:
                return await self._generate_contextual_mock_response(user_id, context, user_message, personality_profile, health_profile)

            # Prepare context for the agent
            agent_context = self._prepare_agent_context(personality_profile, health_profile, context)

            # Get response from appropriate agent
            response = self.letta_client.agents.messages.create(
                agent_id=agent_id,
                messages=[{"role": "user", "content": f"Context: {agent_context}\n\nUser message: {user_message}"}]
            )

            # Extract and format response
            return self._format_agent_response(response, personality_profile, health_profile)

        except Exception as e:
            logger.error(f"Error getting contextual coaching: {str(e)}")
            return {"error": str(e)}

    def _calculate_consistency_improvement(self, vocal_metrics: Dict[str, Any]) -> float:
        """Calculate consistency improvement score"""
        # Implementation for consistency calculation
        jitter = vocal_metrics.get('jitter', 0.015)
        shimmer = vocal_metrics.get('shimmer', 0.020)

        # Lower jitter and shimmer indicate better consistency
        consistency_score = max(0, 10 - (jitter * 500 + shimmer * 300))
        return min(10, consistency_score)

    def _calculate_technique_advancement(self, vocal_metrics: Dict[str, Any]) -> float:
        """Calculate technique advancement score"""
        # Implementation for technique advancement
        vibrato_rate = vocal_metrics.get('vibrato_rate', 5.5)

        # Optimal vibrato rate is around 5-6 Hz
        optimal_vibrato = 5.5
        vibrato_score = max(0, 10 - abs(vibrato_rate - optimal_vibrato) * 2)

        return min(10, vibrato_score)

    def _detect_breakthrough_moments(self, vocal_metrics: Dict[str, Any], personality_profile: VocalPersonalityProfile) -> Dict[str, Any]:
        """Detect breakthrough moments in vocal development"""
        # Implementation for breakthrough detection
        current_score = self._calculate_consistency_improvement(vocal_metrics)

        has_breakthrough = current_score > personality_profile.evolution_score + 2.0

        return {
            "has_breakthrough": has_breakthrough,
            "breakthrough_type": "consistency" if has_breakthrough else None,
            "improvement_delta": current_score - personality_profile.evolution_score
        }

    def _adapt_personality_type(self, vocal_metrics: Dict[str, Any], personality_profile: VocalPersonalityProfile) -> Dict[str, Any]:
        """Adapt personality type based on vocal development patterns"""
        # Implementation for personality adaptation
        return {
            "current_type": personality_profile.personality_type.value,
            "suggested_type": personality_profile.personality_type.value,  # Simplified for now
            "adaptation_confidence": 0.75
        }

    def _analyze_strain_indicators(self, vocal_metrics: Dict[str, Any]) -> List[str]:
        """Analyze vocal strain indicators"""
        indicators = []

        jitter = vocal_metrics.get('jitter', 0.015)
        shimmer = vocal_metrics.get('shimmer', 0.020)

        if jitter > 0.020:
            indicators.append("elevated_jitter")
        if shimmer > 0.025:
            indicators.append("elevated_shimmer")

        return indicators

    def _assess_recovery_needs(self, vocal_metrics: Dict[str, Any], health_profile: VocalHealthProfile) -> Dict[str, Any]:
        """Assess vocal recovery needs"""
        strain_indicators = self._analyze_strain_indicators(vocal_metrics)

        if len(strain_indicators) > 1:
            return {
                "recovery_needed": True,
                "suggested_rest_duration": "2-4 hours",
                "recovery_activities": ["vocal rest", "hydration", "gentle humming"]
            }

        return {
            "recovery_needed": False,
            "suggested_rest_duration": "30 minutes",
            "recovery_activities": ["light vocalization"]
        }

    def _assess_environmental_impact(self, environmental_data: Dict[str, Any], health_profile: VocalHealthProfile) -> Dict[str, Any]:
        """Assess environmental impact on vocal health"""
        if not environmental_data:
            return {"impact_level": "unknown"}

        # Simplified environmental assessment
        return {
            "impact_level": "low",
            "factors": ["humidity", "temperature"],
            "recommendations": ["maintain hydration"]
        }

    def _calculate_risk_level(self, vocal_metrics: Dict[str, Any], health_profile: VocalHealthProfile) -> HealthRiskLevel:
        """Calculate current vocal health risk level"""
        strain_indicators = self._analyze_strain_indicators(vocal_metrics)

        if len(strain_indicators) >= 2:
            return HealthRiskLevel.MODERATE
        elif len(strain_indicators) == 1:
            return HealthRiskLevel.LOW
        else:
            return HealthRiskLevel.LOW

    def _generate_health_recommendations(self, vocal_metrics: Dict[str, Any], health_profile: VocalHealthProfile) -> List[str]:
        """Generate health-based recommendations"""
        recommendations = []
        strain_indicators = self._analyze_strain_indicators(vocal_metrics)

        if "elevated_jitter" in strain_indicators:
            recommendations.append("Focus on breath support exercises")
        if "elevated_shimmer" in strain_indicators:
            recommendations.append("Practice gentle glissandos")

        recommendations.append("Maintain adequate hydration")
        return recommendations

    def _select_agent_for_context(self, context: str) -> Optional[str]:
        """Select appropriate agent based on context"""
        context_mapping = {
            "personality": "personality_coach",
            "health": "health_monitor",
            "patterns": "pattern_analyst",
            "coaching": "personality_coach"
        }

        agent_key = context_mapping.get(context, "personality_coach")
        return self.agents.get(agent_key)

    def _prepare_agent_context(self, personality_profile: VocalPersonalityProfile, health_profile: VocalHealthProfile, context: str) -> str:
        """Prepare context for agent"""
        return f"""
        User Personality Type: {personality_profile.personality_type.value}
        Evolution Score: {personality_profile.evolution_score}
        Health Risk Level: {health_profile.current_risk_level.value}
        Context: {context}
        """

    def _format_agent_response(self, response: Any, personality_profile: VocalPersonalityProfile, health_profile: VocalHealthProfile) -> Dict[str, Any]:
        """Format agent response"""
        # Extract response from Letta agent
        message_content = ""
        for msg in response.messages:
            if msg.message_type == "assistant_message":
                message_content = msg.content
                break

        return {
            "message": message_content,
            "personality_context": personality_profile.personality_type.value,
            "health_context": health_profile.current_risk_level.value,
            "suggestions": ["Continue practice", "Focus on technique"]
        }

    async def _update_personality_profile(self, user_id: str, evolution_data: Dict[str, Any]):
        """Update personality profile with evolution data"""
        if not self.supabase:
            return

        try:
            # Update personality profile in database
            self.supabase.table('vocal_personality_profiles').update({
                'evolution_score': evolution_data.get('consistency_improvement', 0),
                'total_evolution_points': evolution_data.get('technique_advancement', 0),
                'updated_at': datetime.now(pytz.utc).isoformat()
            }).eq('user_id', user_id).execute()

        except Exception as e:
            logger.error(f"Error updating personality profile: {str(e)}")

    async def _update_health_profile(self, user_id: str, health_analysis: Dict[str, Any]):
        """Update health profile with analysis data"""
        if not self.supabase:
            return

        try:
            # Update health profile in database
            self.supabase.table('vocal_health_profiles').update({
                'current_risk_level': health_analysis.get('risk_level', HealthRiskLevel.LOW).value,
                'last_health_check': datetime.now(pytz.utc).isoformat(),
                'updated_at': datetime.now(pytz.utc).isoformat()
            }).eq('user_id', user_id).execute()

        except Exception as e:
            logger.error(f"Error updating health profile: {str(e)}")

    async def _generate_contextual_mock_response(self, user_id: str, context: str, user_message: str, personality_profile: VocalPersonalityProfile, health_profile: VocalHealthProfile) -> Dict[str, Any]:
        """Generate contextual mock response when AI is not available"""
        return {
            "message": f"Based on your {personality_profile.personality_type.value} personality and {health_profile.current_risk_level.value} health status, here's my advice: {user_message}",
            "suggestions": ["Practice regularly", "Monitor your progress"],
            "personality_context": personality_profile.personality_type.value,
            "health_context": health_profile.current_risk_level.value
        }

    def _get_fallback_personality_profile(self, user_id: str) -> VocalPersonalityProfile:
        """Fallback personality profile when database is unavailable"""
        return VocalPersonalityProfile(
            user_id=user_id,
            personality_type=VocalPersonalityType.ANALYTICAL,
            evolution_score=1.0,
            breakthrough_moments=[],
            learning_preferences={"style": "visual", "pace": "moderate"},
            coaching_adaptations={"tone": "encouraging", "detail_level": "medium"},
            vocal_fingerprint={},
            last_evolution_date=datetime.now(pytz.utc),
            total_evolution_points=0,
            created_at=datetime.now(pytz.utc),
            updated_at=datetime.now(pytz.utc)
        )

    def _get_fallback_health_profile(self, user_id: str) -> VocalHealthProfile:
        """Fallback health profile when database is unavailable"""
        return VocalHealthProfile(
            user_id=user_id,
            current_risk_level=HealthRiskLevel.LOW,
            strain_patterns={},
            recovery_patterns={},
            environmental_correlations={},
            warning_indicators=[],
            optimal_practice_windows=[],
            health_trends={},
            last_health_check=datetime.now(pytz.utc),
            created_at=datetime.now(pytz.utc),
            updated_at=datetime.now(pytz.utc)
        )

    def _parse_datetime_with_tz(self, datetime_str: str) -> datetime:
        """Parse datetime string and ensure it's timezone-aware"""
        dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            # If no timezone info, assume UTC
            dt = pytz.utc.localize(dt)
        return dt

class VocalCoachingAdvisor:
    """Combines vocal coaching, feedback, and session recommendations into unified analysis logic"""

    @staticmethod
    def generate_integrated_recommendations(
        personality_result: Dict[str, Any],
        health_result: Dict[str, Any]
    ) -> List[str]:
        """Generate integrated recommendations based on both personality and health analysis"""
        recommendations = []

        # Personality-based recommendations
        if personality_result.get("breakthrough_detection", {}).get("has_breakthrough"):
            recommendations.append("🎉 Breakthrough detected! Continue with current practice intensity")

        # Health-based recommendations
        if health_result.get("recovery_needs", {}).get("recovery_needed"):
            recommendations.append("⚠️ Vocal rest recommended - limit practice to light exercises")

        # Combined recommendations
        recommendations.extend([
            "📈 Your vocal personality is evolving - stay consistent with practice",
            "🎯 Focus on breath support for optimal vocal health",
            "💡 Consider practicing during your optimal time windows"
        ])

        return recommendations

    @staticmethod
    def generate_next_session_guidance(
        personality_result: Dict[str, Any],
        health_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate guidance for the next practice session"""
        return {
            "recommended_duration": "15-20 minutes" if health_result.get("strain_indicators") else "20-30 minutes",
            "focus_areas": ["breath support", "vocal consistency"],
            "exercises": ["lip trills", "gentle scales", "sustained tones"],
            "cautions": health_result.get("recommendations", [])
        }

    @staticmethod
    def generate_dashboard_recommendations(
        personality_profile: VocalPersonalityProfile,
        health_profile: VocalHealthProfile
    ) -> List[str]:
        """Generate recommendations for dashboard display"""
        recommendations = []

        if personality_profile.evolution_score < 3.0:
            recommendations.append("Focus on consistency to boost your vocal evolution")

        if health_profile.current_risk_level == HealthRiskLevel.MODERATE:
            recommendations.append("Consider reducing practice intensity temporarily")

        recommendations.append("Your personalized coaching is adapting to your progress")

        return recommendations