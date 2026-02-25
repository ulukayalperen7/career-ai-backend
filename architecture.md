# Agentic Career Assistant AI Architecture

```mermaid
graph TD
    User[Recruiter] -->|Message| API[FastAPI Backend POST /chat]
    
    subgraph Backend
        API --> Memory[(Sliding Window Memory)]
        API --> Primary[Primary Agent]
        
        subgraph Agent Loop
            Primary -->|Draft| Critic[Critic Agent]
            Critic -->|Score < 8 & Feedback| Primary
        end
        
        Critic -->|Score >= 8| API
    end
    
    subgraph Knowledge Base
        Profile[profile.txt CV Context] --> Primary
    end
    
    subgraph External Tools
        Pushover[Pushover API Notifications]
        Human[Alperen Manual Intervention]
    end
    
    API -->|Final Response| User
    API -->|Notify New Message / Response| Pushover
    Primary -->|[NEEDS_HUMAN]| Human
    Human --> Pushover
```
