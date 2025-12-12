from game import UltimateTicTacToe
from agent import QLearningAgent, RandomAgent

def play_match(agent1, agent2, games=100):
    """Play matches between two agents"""
    game = UltimateTicTacToe()
    results = {'agent1_wins': 0, 'agent2_wins': 0, 'draws': 0}

    for _ in range(games):
        state = game.reset()
        done = False

        while not done:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            if game.current_player == 1:
                action = agent1.get_action(state, legal_moves, training=False)
            else:
                action = agent2.get_action(state, legal_moves, training=False)

            state, reward, done = game.make_move(*action)

        if game.winner == 1:
            results['agent1_wins'] += 1
        elif game.winner == 2:
            results['agent2_wins'] += 1
        else:
            results['draws'] += 1

    return results

def evaluate():
    # Load trained agent
    trained_agent = QLearningAgent()
    trained_agent.load('trained_agent.pkl')
    trained_agent.epsilon = 0  # No exploration

    # Test against random
    random_agent = RandomAgent()

    print("Trained Agent (Player 1) vs Random (Player 2):")
    results = play_match(trained_agent, random_agent, games=100)
    print(f"Wins: {results['agent1_wins']}, Losses: {results['agent2_wins']}, Draws: {results['draws']}")
    print(f"Win Rate: {results['agent1_wins']/100*100:.1f}%")

    print("\nRandom (Player 1) vs Trained Agent (Player 2):")
    results = play_match(random_agent, trained_agent, games=100)
    print(f"Wins: {results['agent1_wins']}, Losses: {results['agent2_wins']}, Draws: {results['draws']}")
    print(f"Trained Win Rate: {results['agent2_wins']/100*100:.1f}%")

if __name__ == "__main__":
    evaluate()