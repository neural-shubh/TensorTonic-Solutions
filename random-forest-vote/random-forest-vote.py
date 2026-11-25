def random_forest_vote(predictions):
  results = []

  for i in range(len(predictions[0])):
    sample_votes = []

    for tree in predictions:
      sample_votes.append(tree[i])

    votes = {}

    for vote in sample_votes:
      votes[vote] = votes.get(vote, 0) + 1

    winner = max(votes, key = lambda x: (votes[x], -x)) 
    results.append(winner)

  return results