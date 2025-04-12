The goal of this project is to train a model, using RL, to efficiently search through a large dataset of emails to answer a user's query. The idea is that an agent so trained could be eg. exposed through a Gmail plugin, and let users ask natural-language queries such as "what time does my wife's flight arrive on Friday" or "what are the next steps I committed to for project X", and let the agent search the email to find relevant results.

## Data

The agent is both trained and validated on the Enron email dataset. To generate training data, we iterated over the email inboxes of several Enron employees. For each one, we fed their emails into a prompt and asked the LLM to come up with a list of plausible questions that the emails collectively answer. We then used these questions to train the agent.
