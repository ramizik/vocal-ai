# VocalAIAgent - AI-Powered Vocal Coaching System 🎤

**Winner of "Most Ambitious Vapi Project" at UC Berkeley Hackathon 2025**

VocalAIAgent is a comprehensive AI-powered vocal coaching system that combines real-time voice analysis, intelligent coaching, and conversational AI agents to create a holistic vocal development experience. Built during the world's biggest AI in-person hackathon where over 1,200 developers competed.

YouTube Demo:

[![Vocal AI](https://img.youtube.com/vi/4XWiMuE9wwM/0.jpg)](https://youtu.be/4XWiMuE9wwM)


## 🎯 Problem Statement

Vocal training can be an isolated and inconsistent process. Singers and speakers often struggle with:
- Lack of real-time feedback during practice sessions
- Limited access to personalized coaching based on their specific vocal characteristics
- Difficulty tracking progress and identifying improvement areas
- Fragmented resources across multiple platforms and tools
- Inconsistent practice routines without proper guidance

## Solution

VocalAIAgent addresses these challenges by providing a unified, intelligent coaching platform that combines voice analysis, personalized AI coaching, and comprehensive progress tracking in one seamless experience.

## 🚀 Key Features

### Core Vocal Analysis
- **🎵 Real-Time Pitch Detection**: Instant feedback during practice sessions with live pitch visualization
- **📊 Deep Vocal Analysis**: Advanced metrics including jitter, shimmer, vibrato rate, vocal range analysis
- **🎯 Voice Type Classification**: Automatic classification of voice types (soprano, alto, tenor, bass)
- **📈 Progress Tracking**: Comprehensive tracking of vocal improvements over time

### AI-Powered Coaching System
- **🤖 Dual-AI Architecture**: Proactive Fetch.ai Agent + Reactive Letta Conversational Agent
- **💬 Stateful Conversations**: AI coach that remembers context and discusses specific progress
- **📋 Personalized Lesson Plans**: Dynamic lesson generation based on vocal analysis and user goals
- **🎓 Exercise Recommendations**: Tailored vocal exercises based on analysis results

### Advanced Features
- **🗣️ VAPI Voice Integration**: Real-time voice conversations with AI coach
- **📱 Multimodal Interface**: Support for voice input, text chat, and visual feedback
- **🔄 Lesson Feedback Loop**: Comprehensive storage and analysis of lesson completion data
- **📊 AI-Generated Reports**: Daily summaries of performance trends and insights
- **�� Community Features**: Progress sharing and vocal challenges

### Data & Memory Management
- **💾 Persistent Memory**: User preferences, vocal characteristics, and practice history retention
- **📤 Export Capabilities**: Save vocal analyses, lesson plans, and progress reports
- **🔐 Secure Data Storage**: Supabase integration with proper authentication and RLS
- **�� Session Management**: Comprehensive tracking of practice sessions and improvements

## 🛠 Tech Stack

### Frontend
- **React**: Modern component-based UI framework
- **TypeScript**: Type-safe development with enhanced IDE support
- **Vite**: Fast build tool and development server
- **Tailwind CSS**: Utility-first CSS framework for responsive design
- **Framer Motion**: Smooth animations and transitions

### Backend
- **FastAPI**: High-performance Python web framework with automatic API docs
- **Python**: Core backend language with extensive AI/ML libraries
- **Uvicorn**: ASGI server for production deployment

### AI Services
- **Letta**: Stateful conversational AI with long-term memory capabilities
- **Fetch.ai**: Autonomous agent system for proactive analysis and reporting
- **VAPI**: Voice AI platform for real-time voice conversations

### Database & Authentication
- **Supabase**: PostgreSQL database with built-in authentication and real-time features
- **Row Level Security (RLS)**: Secure user data isolation

### Voice Processing
- **Web Audio API**: Real-time audio processing and pitch detection
- **Custom Voice Analyzer**: Advanced vocal metrics calculation

### Hosting & Deployment
- **Netlify**: Frontend hosting with automatic deployments
- **Google Cloud Run**: Scalable backend container hosting
- **Docker**: Containerized backend for consistent deployments

## 🚀 Getting Started

### Prerequisites
- Node.js (v18 or higher)
- Python (v3.8 or higher)
- Docker (for backend deployment)

### Frontend Setup
```bash
cd src
npm install
npm run dev
```

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

### Environment Variables
Create `.env` files in both frontend and backend directories with the necessary API keys and configuration.

## Why This Matters

Vocal training today lacks the personalized, data-driven approach that modern AI can provide. VocalAIAgent brings together voice science, conversational AI, and personalized coaching into one intelligent system, offering a more effective, engaging, and accessible vocal training experience.

By combining real-time voice analysis, stateful AI conversations, and comprehensive progress tracking, this tool showcases the potential of Generative AI in revolutionizing music education and vocal development.

##  Live Demo

https://prismatic-buttercream-5f0d5a.netlify.app/

---

**Built with ❤️ at UC Berkeley Hackathon 2025**
