from receptor.agents.agent import BaseAgent


class MultiAgentWrap(BaseAgent):
    def __init__(self, agents):
        self.agents = agents

    def act(self, obs):
        # Takes tuple obses for each agent
        # Returns tuple actions for each agent
        # If obs for some agent is None, returns dummy None action for that agent.
        actions = []
        for agent, obs in zip(self.agents, obs):
            act = agent.act(obs) if obs is not None else None
            actions.append(act)
        return tuple(actions)

    def predict_on_batch(self, obs_batch):
        outs = []
        for agent, obs in zip(self.agents, obs_batch):
            out = agent.predict_on_batch(obs) if obs is not None else None
            outs.append(out)
        return tuple(outs)

    def train_on_batch(self, rollout, lr=None, aux_losses=(), summarize=False, importance=None):
        pass

    def explore(self, obs):
        pass
