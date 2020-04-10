# interpret-RL
Using imitation learning and model diffs to interpret reinforcement learning policies

## Environments (in progress)

- 2D GridWorld with obstacles and/or roads
- Recipe or something logic-based?

## Experimental Design (in progress)

Axes of comparison:

1. The policies are completely different, so there's a diff everywhere -> interpretable model is more useful
2.  Both policies are complicated but similar, except in a few key areas -> diff model is more useful (because the interpretable model would have to interpret complicated regions of the blackbox policy that are similar to the behavior policy anyway, and that the diff model would just ignore by telling the user to follow the behavior policy)
3.  Behavior policy is simple and blackbox policy is complex, but they take similar actions -> interpretable model might be more useful, diff model would show many small diffs which might be harder for a user to parse (this feels like the key scenario, but I'm having trouble breaking it down further)
4.  Both policies are simple (easily interpreted) and similar -> unclear which model would be more useful
  - More generally, both policies similar (simple or complex)
5.  Complicated action space (many possible actions) -> unclear which model would be more useful
6. Complicated state space -> unclear which model would be more useful
