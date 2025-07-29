# This code demonstrates how to build a natural language interface for querying a SQL database 
# using LangChain and OpenAI’s language model. It loads a SQLite database of customers,
# initializes an OpenAI LLM, and creates an agent that translates plain English questions into SQL queries, 
# executes them on the database, and returns human-readable answers. 

# For example, when asked “Which customers are from California?”, 
# the agent generates the appropriate SQL query, runs it, and outputs the matching customer names. 
# This approach works well for small databases where the schema can be included in the prompt, 
# but for larger, more complex databases, integrating retrieval methods with vector stores is 
# recommended to efficiently handle schema information. Overall, it enables users to interact with 
# databases conversationally without needing to write SQL themselves.

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


# NOTE

"""
For a small DB (like customers.db), injecting all schema works.

But to scale to Large/Complex Schemas, for real-world systems (ERP, CRM, etc.), 
use RAG to retrive schema from Vector databases like FAISS/Chroma/Pinecone to enable Semantic Search for Relevant Schema
"""
