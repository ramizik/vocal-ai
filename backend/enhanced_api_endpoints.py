"""
Enhanced API endpoints for Progressive Vocal Personality Development and Health Monitoring
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import pytz
from enhanced_letta_service import EnhancedLettaService, VocalPersonalityType, HealthRiskLevel, VocalCoachingAdvisor

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize enhanced service
enhanced_letta = EnhancedLettaService()

# Pydantic models for request/response
class VocalAnalysisRequest(BaseModel):
    user_id: str
    vocal_metrics: Dict[str, Any]
    environmental_data: Optional[Dict[str, Any]] = None

class PersonalityEvolutionRequest(BaseModel):
    user_id: str
    vocal_metrics: Dict[str, Any]
    practice_duration: Optional[int] = None
    breakthrough_notes: Optional[str] = None

class ContextualCoachingRequest(BaseModel):
    user_id: str
    context: str  # 'personality', 'health', 'patterns', 'coaching'
    message: str
    vocal_data: Optional[Dict[str, Any]] = None

class PersonalityProfileResponse(BaseModel):
    user_id: str
    personality_type: str
    evolution_score: float
    total_evolution_points: int
    breakthrough_count: int
    learning_preferences: Dict[str, Any]
    coaching_adaptations: Dict[str, Any]

class HealthProfileResponse(BaseModel):
    user_id: str
    current_risk_level: str
    strain_indicators: List[str]
    recovery_recommendations: List[str]
    optimal_practice_times: List[str]
    health_trends: Dict[str, Any]

@router.post("/api/letta/personality/analyze")
async def analyze_vocal_evolution(request: PersonalityEvolutionRequest):
    """Analyze vocal evolution and personality development"""
    try:
        evolution_data = await enhanced_letta.analyze_vocal_evolution(
            user_id=request.user_id,
            vocal_metrics=request.vocal_metrics
        )

        return {
            "success": True,
            "data": evolution_data,
            "message": "Vocal evolution analysis completed"
        }
    except Exception as e:
        logger.error(f"Error in personality analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/letta/health/monitor")
async def monitor_vocal_health(request: VocalAnalysisRequest):
    """Monitor vocal health and provide early warning system"""
    try:
        health_analysis = await enhanced_letta.monitor_vocal_health(
            user_id=request.user_id,
            vocal_metrics=request.vocal_metrics,
            environmental_data=request.environmental_data
        )

        return {
            "success": True,
            "data": health_analysis,
            "message": "Vocal health monitoring completed"
        }
    except Exception as e:
        logger.error(f"Error in health monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/letta/coaching/contextual")
async def get_contextual_coaching(request: ContextualCoachingRequest):
    """Get contextual coaching based on personality and health profiles"""
    try:
        coaching_response = await enhanced_letta.get_contextual_coaching(
            user_id=request.user_id,
            context=request.context,
            user_message=request.message
        )

        return {
            "success": True,
            "data": coaching_response,
            "message": "Contextual coaching provided"
        }
    except Exception as e:
        logger.error(f"Error in contextual coaching: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/letta/personality/profile/{user_id}")
async def get_personality_profile(user_id: str) -> PersonalityProfileResponse:
    """Get user's vocal personality profile"""
    try:
        profile = await enhanced_letta.get_personality_profile(user_id)

        return PersonalityProfileResponse(
            user_id=profile.user_id,
            personality_type=profile.personality_type.value,
            evolution_score=profile.evolution_score,
            total_evolution_points=profile.total_evolution_points,
            breakthrough_count=len(profile.breakthrough_moments),
            learning_preferences=profile.learning_preferences,
            coaching_adaptations=profile.coaching_adaptations
        )
    except Exception as e:
        logger.error(f"Error getting personality profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/letta/health/profile/{user_id}")
async def get_health_profile(user_id: str) -> HealthProfileResponse:
    """Get user's vocal health profile"""
    try:
        profile = await enhanced_letta.get_health_profile(user_id)

        return HealthProfileResponse(
            user_id=profile.user_id,
            current_risk_level=profile.current_risk_level.value,
            strain_indicators=[],  # Simplified for response
            recovery_recommendations=[],  # Simplified for response
            optimal_practice_times=[],  # Simplified for response
            health_trends=profile.health_trends
        )
    except Exception as e:
        logger.error(f"Error getting health profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/letta/integrated-analysis")
async def integrated_vocal_analysis(
    request: VocalAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Comprehensive integrated analysis combining personality evolution and health monitoring
    This is the main endpoint that should be called after each practice session
    """
    try:
        # Run personality and health analysis in parallel
        import asyncio

        personality_task = enhanced_letta.analyze_vocal_evolution(
            user_id=request.user_id,
            vocal_metrics=request.vocal_metrics
        )

        health_task = enhanced_letta.monitor_vocal_health(
            user_id=request.user_id,
            vocal_metrics=request.vocal_metrics,
            environmental_data=request.environmental_data
        )

        # Wait for both analyses to complete
        personality_result, health_result = await asyncio.gather(
            personality_task, health_task
        )

        # Generate integrated insights
        integrated_insights = {
            "personality_evolution": personality_result,
            "health_monitoring": health_result,
            "integrated_recommendations": VocalCoachingAdvisor.generate_integrated_recommendations(
                personality_result, health_result
            ),
            "next_session_guidance": VocalCoachingAdvisor.generate_next_session_guidance(
                personality_result, health_result
            )
        }

        return {
            "success": True,
            "data": integrated_insights,
            "message": "Integrated vocal analysis completed"
        }

    except Exception as e:
        logger.error(f"Error in integrated analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/letta/dashboard/insights/{user_id}")
async def get_dashboard_insights(user_id: str):
    """Get insights for dashboard display"""
    try:
        # Get both profiles
        personality_profile = await enhanced_letta.get_personality_profile(user_id)
        health_profile = await enhanced_letta.get_health_profile(user_id)

        # Use timezone-aware datetime for calculations
        now_utc = datetime.now(pytz.utc)

        # Handle timezone conversion for created_at
        if personality_profile.created_at.tzinfo is None:
            # If created_at is naive, assume it's UTC
            created_at_utc = pytz.utc.localize(personality_profile.created_at)
        else:
            # If it's already timezone-aware, convert to UTC
            created_at_utc = personality_profile.created_at.astimezone(pytz.utc)

        days_training = max(1, (now_utc - created_at_utc).days)

        dashboard_data = {
            "personality_summary": {
                "type": personality_profile.personality_type.value,
                "evolution_score": personality_profile.evolution_score,
                "days_training": days_training,
                "insights_learned": personality_profile.total_evolution_points,
                "adaptation_score": personality_profile.evolution_score
            },
            "health_summary": {
                "risk_level": health_profile.current_risk_level.value,
                "strain_indicators": len(health_profile.warning_indicators),
                "optimal_windows": len(health_profile.optimal_practice_windows),
                "last_check": health_profile.last_health_check.isoformat() if health_profile.last_health_check else now_utc.isoformat()
            },
            "recommendations": VocalCoachingAdvisor.generate_dashboard_recommendations(
                personality_profile, health_profile
            )
        }

        return {
            "success": True,
            "data": dashboard_data,
            "message": "Dashboard insights retrieved"
        }

    except Exception as e:
        logger.error(f"Error getting dashboard insights: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))