#!/usr/bin/env python3
"""
Evaluation script for trained LLM models
Demonstrates model performance on Game7 environment
"""

import os
import sys
import yaml
import argparse
import logging
import torch
import numpy as np
from typing import Dict, Any, List
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.llm import AutoModel
from environments.GymEnv import Game7Env

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_model(model_path: str, model_config: Dict[str, Any]) -> AutoModel:
    """Load trained model"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    agent = AutoModel(
        model_name_or_path=model_config['model_name_or_path'],
        ckpt_path=model_path,
        quantization=model_config.get('quantization'),
        generation=model_config.get('generation'),
        device=device
    )
    
    return agent

def play_game_demo(agent: AutoModel, env: Game7Env, num_games: int = 5) -> List[Dict[str, Any]]:
    """Play demonstration games"""
    game_results = []
    
    for game in range(num_games):
        logging.info(f"\n=== Game {game + 1} ===")
        
        state = env.reset()
        done = False
        step_count = 0
        game_history = []
        
        while not done and step_count < 20:
            # Display current state
            env.render()
            
            # Get model's action
            action_info = agent.act(state, deterministic=True)
            action = action_info['action']
            response = action_info['response']
            
            logging.info(f"Model response: '{response}'")
            logging.info(f"Model action: {action} (adds {action + 1})")
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Record step
            step_info = {
                'step': step_count,
                'current_sum': info.get('current_sum', 0),
                'action': action,
                'action_value': action + 1,
                'response': response,
                'reward': reward,
                'done': done,
                'info': info
            }
            game_history.append(step_info)
            
            # Opponent's turn (if game continues)
            if not done:
                opponent_move = info.get('opponent_move', 0)
                logging.info(f"Opponent adds: {opponent_move}")
                logging.info(f"New total: {info.get('current_sum_after_opponent', 0)}/7")
            
            state = next_state
            step_count += 1
        
        # Game result
        final_reward = sum(step['reward'] for step in game_history)
        result = "WIN" if final_reward > 0 else "LOSS"
        logging.info(f"Game {game + 1} Result: {result} (Reward: {final_reward})")
        
        game_results.append({
            'game_id': game + 1,
            'result': result,
            'final_reward': final_reward,
            'steps': step_count,
            'history': game_history
        })
    
    return game_results

def evaluate_model_performance(agent: AutoModel, env: Game7Env, num_episodes: int = 100) -> Dict[str, float]:
    """Evaluate model performance with detailed metrics"""
    total_rewards = []
    episode_lengths = []
    win_count = 0
    action_distribution = {0: 0, 1: 0, 2: 0}  # Count of each action
    
    logging.info(f"Evaluating model on {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done and step_count < 20:
            action_info = agent.act(state, deterministic=True)
            action = action_info['action']
            
            # Record action
            if action in action_distribution:
                action_distribution[action] += 1
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            step_count += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        if episode_reward > 0:
            win_count += 1
    
    # Calculate metrics
    metrics = {
        'avg_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'avg_length': np.mean(episode_lengths),
        'win_rate': win_count / num_episodes,
        'min_reward': np.min(total_rewards),
        'max_reward': np.max(total_rewards),
        'total_games': num_episodes,
        'wins': win_count,
        'losses': num_episodes - win_count
    }
    
    # Action distribution
    total_actions = sum(action_distribution.values())
    if total_actions > 0:
        for action, count in action_distribution.items():
            metrics[f'action_{action}_rate'] = count / total_actions
    
    return metrics

def save_evaluation_results(results: Dict[str, Any], game_demos: List[Dict[str, Any]], output_path: str):
    """Save evaluation results to file"""
    output_data = {
        'metrics': results,
        'game_demos': game_demos,
        'timestamp': str(datetime.now())
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logging.info(f"Evaluation results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained LLM model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to model configuration file')
    parser.add_argument('--num-games', type=int, default=5, help='Number of demonstration games')
    parser.add_argument('--num-episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Output file path')
    parser.add_argument('--demo-only', action='store_true', help='Only run demonstration games')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
    env = Game7Env(seed=42)
    
    # Load model
    logging.info(f"Loading model from {args.model_path}")
    agent = load_model(args.model_path, config['model'])
    
    if args.demo_only:
        # Run demonstration games only
        game_demos = play_game_demo(agent, env, args.num_games)
        results = {'demo_games': len(game_demos)}
    else:
        # Run demonstration games
        logging.info("Running demonstration games...")
        game_demos = play_game_demo(agent, env, args.num_games)
        
        # Run performance evaluation
        logging.info("Running performance evaluation...")
        results = evaluate_model_performance(agent, env, args.num_episodes)
        
        # Display results
        logging.info("\n=== Evaluation Results ===")
        for key, value in results.items():
            if isinstance(value, float):
                logging.info(f"{key}: {value:.4f}")
            else:
                logging.info(f"{key}: {value}")
    
    # Save results
    save_evaluation_results(results, game_demos, args.output)
    
    logging.info("Evaluation completed!")

if __name__ == "__main__":
    main() 