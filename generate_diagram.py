from diagrams import Diagram, Cluster, Edge
from diagrams.programming.language import Python
from diagrams.onprem.client import User
from diagrams.onprem.network import Internet
from diagrams.custom import Custom
from diagrams.generic.database import SQL

with Diagram("Agentic Career Assistant AI Architecture", show=False, direction="LR"):
    recruiter = User("Recruiter")
    
    with Cluster("FastAPI Backend (main.py)"):
        api = Python("POST /chat")
        memory = SQL("Sliding Window Memory")
        
        with Cluster("Agent Loop"):
            primary_agent = Python("Primary Agent\n(Drafts Response)")
            critic_agent = Python("Critic Agent\n(Evaluates Draft)")
            
            primary_agent >> Edge(label="Draft") >> critic_agent
            critic_agent >> Edge(label="Score < 8\nFeedback") >> primary_agent
            
    with Cluster("External Tools (tools.py)"):
        pushover = Internet("Pushover API\n(Notifications)")
        human_intervention = User("Alperen\n(Manual Intervention)")
        
    with Cluster("Knowledge Base"):
        profile = Python("profile.txt\n(CV Context)")
        
    recruiter >> Edge(label="Message") >> api
    api >> memory
    api >> primary_agent
    profile >> primary_agent
    
    critic_agent >> Edge(label="Score >= 8") >> api
    api >> Edge(label="Final Response") >> recruiter
    
    api >> Edge(label="Notify New Message / Response") >> pushover
    primary_agent >> Edge(label="[NEEDS_HUMAN]") >> human_intervention
    human_intervention >> pushover
