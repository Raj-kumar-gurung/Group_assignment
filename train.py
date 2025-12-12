import numpy as np
import matplotlib.pyplot as plt
from game import UltimateTicTacToe
from agent import QLearningAgent, RandomAgent

def train(episodes=10000):
    game = UltimateTicTacToe()
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=1.0)

    wins = []
    draws = []
    losses = []

    print("Training via self-play...")

    for episode in range(episodes):
        state = game.reset()
        done = False

        while not done:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                break

            action = agent.get_action(state, legal_moves, training=True)
            next_state, reward, done = game.make_move(*action)

            if not done:
                next_legal_moves = game.get_legal_moves()
                agent.update(state, action, reward, next_state, next_legal_moves, done)
            else:
                agent.update(state, action, reward, next_state, [], done)

            state = next_state

        # Track results
        if game.winner == 1:
            wins.append(1)
        elif game.winner == 0:
            draws.append(1)
        else:
            losses.append(1)

        agent.decay_epsilon()

        if (episode + 1) % 1000 == 0:
            recent = 1000
            w = sum(wins[-recent:])
            d = sum(draws[-recent:])
            l = sum(losses[-recent:])
            print(f"Episode {episode + 1}: W:{w} D:{d} L:{l} Eps:{agent.epsilon:.3f}")

    agent.save('trained_agent.pkl')
    plot_results(wins, draws, losses)
    return agent

def plot_results(wins, draws, losses):
    window = 100
    x = range(window, len(wins) + 1)
    w = [sum(wins[i-window:i])/window for i in x]
    d = [sum(draws[i-window:i])/window for i in x]
    l = [sum(losses[i-window:i])/window for i in x]

    plt.figure(figsize=(10, 6))
    plt.plot(x, w, label='Wins')
    plt.plot(x, d, label='Draws')
    plt.plot(x, l, label='Losses')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (100-episode window)')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    train(episodes=10000)