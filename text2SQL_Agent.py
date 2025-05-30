# pip install langchain openai sqlalchemy sqlite3
# export OPENAI_API_KEY=your_key_here

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase

# Load database
db = SQLDatabase.from_uri("sqlite:///customers.db")

# Initialize LLM
llm = OpenAI(temperature=0)

# Create toolkit and agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type="openai-tools"
)

# Run natural language question
query = "Which customers are from California?"
response = agent_executor.run(query)

print("\n=== SQL Generated and Result ===")
print(response)

# OUTPUT

"""
> Entering new AgentExecutor chain...
Thought: I need to query the customers table where state is 'California'.
Action: sql_db_query
Action Input: SELECT * FROM customers WHERE state = 'California';
Observation: [(1, 'Alice', 'alice@example.com', 'California'), (3, 'Charlie', 'charlie@example.com', 'California')]
Final Answer: Alice and Charlie are customers from California.
"""
