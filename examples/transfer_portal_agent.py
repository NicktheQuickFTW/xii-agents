#!/usr/bin/env python3
"""
Transfer Portal Agent with Memory and ML
Purpose: Autonomously monitors college transfer portal data, learns patterns, and makes predictions
Version: 1.0.0

Usage:
  Run directly: python transfer_portal_agent.py
  Or schedule: python ../utils/autonomous_scheduler.py add transfer_portal ./transfer_portal_agent.py --type interval --interval 3600
  
Requirements:
  - Python 3.8+
  - Install dependencies: pip install requests beautifulsoup4 pandas scikit-learn
"""

import os
import sys
import time
import json
import random
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import traceback

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import autonomous agent base class
try:
    from utils.autonomous_agent import AutonomousAgent
except ImportError:
    print("Error: Could not import AutonomousAgent. Make sure you're running from the correct directory.")
    sys.exit(1)

# Try importing optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

class TransferPortalAgent(AutonomousAgent):
    """
    Agent that monitors college transfer portal data, learns patterns, and makes predictions
    about player transfers in college athletics.
    """
    
    def __init__(
        self, 
        agent_id: str = "transfer_portal",
        memory_type: str = "sqlite", 
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the transfer portal agent"""
        # Default configuration
        default_config = {
            "refresh_interval": 3600,  # 1 hour
            "sources": ["on3", "247sports", "rivals"],
            "sports": ["football", "basketball"],
            "max_players_per_request": 100,
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # Merge with provided config
        if config:
            default_config.update(config)
        
        # Initialize base class
        super().__init__(agent_id, memory_type, enable_learning=True, config=default_config)
        
        # Check for required dependencies
        if not BS4_AVAILABLE:
            self.logger.warning("BeautifulSoup not installed. Web scraping capabilities will be limited.")
        
        if not PANDAS_AVAILABLE:
            self.logger.warning("Pandas not installed. Data analysis capabilities will be limited.")
        
        # Initialize data storage
        self.players = {}  # Current players in the portal
        self.last_refresh = None
        
        # Set up ML models if learning is available
        if self.learning:
            self._setup_models()
            
            # Load existing models if available
            self.load_models()
    
    def _setup_models(self) -> None:
        """Set up machine learning models for predictions"""
        # Create a model for predicting commitment likelihood
        self.learning.create_model(
            "commitment_predictor", 
            "random_forest_classifier",
            {"n_estimators": 100, "max_depth": 10}
        )
        
        # Create a model for predicting player rating
        self.learning.create_model(
            "rating_predictor",
            "random_forest_regressor",
            {"n_estimators": 100, "max_depth": 15}
        )
    
    def _fetch_players(self, source: str, sport: str) -> List[Dict[str, Any]]:
        """
        Fetch players from a specific source and sport
        
        Note: In a real implementation, this would use actual APIs or web scraping.
        This example generates simulated data.
        """
        # In a real implementation, this would use requests to fetch data from source
        if not BS4_AVAILABLE:
            self.logger.warning(f"BeautifulSoup not installed. Using mock data for {source}/{sport}.")
            return self._generate_mock_data(source, sport, 20)
        
        # Simulate fetching from different sources
        if source == "on3":
            return self._generate_mock_data(source, sport, random.randint(15, 30))
        elif source == "247sports":
            return self._generate_mock_data(source, sport, random.randint(20, 40))
        elif source == "rivals":
            return self._generate_mock_data(source, sport, random.randint(10, 25))
        else:
            self.logger.warning(f"Unknown source: {source}")
            return []
    
    def _generate_mock_data(self, source: str, sport: str, count: int) -> List[Dict[str, Any]]:
        """Generate mock player data for demonstration purposes"""
        players = []
        
        # Schools and positions based on sport
        schools = [
            "Alabama", "Georgia", "Ohio State", "Michigan", "Clemson", 
            "Texas", "Oklahoma", "LSU", "Notre Dame", "USC",
            "Oregon", "Florida", "Miami", "Penn State", "Texas A&M"
        ]
        
        if sport == "football":
            positions = ["QB", "RB", "WR", "TE", "OL", "DL", "LB", "CB", "S", "K"]
        else:  # basketball
            positions = ["PG", "SG", "SF", "PF", "C"]
        
        # First names and last names for generating random names
        first_names = [
            "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles",
            "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua"
        ]
        
        last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
            "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
        ]
        
        for i in range(count):
            # Generate a random player
            player = {
                "name": f"{random.choice(first_names)} {random.choice(last_names)}",
                "position": random.choice(positions),
                "previous_school": random.choice(schools),
                "eligibility": random.choice(["Freshman", "Sophomore", "Junior", "Senior", "Graduate"]),
                "rating": round(random.uniform(5.0, 10.0), 1),
                "stars": random.randint(1, 5),
                "height": random.randint(65, 79) if sport == "football" else random.randint(70, 84),
                "weight": random.randint(170, 320) if sport == "football" else random.randint(180, 280),
                "hometown": f"City, State",
                "offers": random.randint(1, 15),
                "visits": random.randint(0, 5),
                "last_updated": (datetime.now() - timedelta(days=random.randint(0, 10))).isoformat(),
                "source": source,
                "sport": sport,
                "committed": random.random() > 0.7,  # 30% chance of being uncommitted
                "commitment_school": None
            }
            
            # If committed, add a commitment school
            if player["committed"]:
                available_schools = [s for s in schools if s != player["previous_school"]]
                player["commitment_school"] = random.choice(available_schools)
            
            players.append(player)
        
        return players
    
    def refresh_data(self) -> bool:
        """Refresh player data from all sources"""
        self.logger.info("Refreshing transfer portal data")
        
        new_players = {}
        
        # Fetch data from each source and sport
        for source in self.config["sources"]:
            for sport in self.config["sports"]:
                try:
                    self.logger.info(f"Fetching data from {source} for {sport}")
                    players = self._fetch_players(source, sport)
                    
                    # Process each player
                    for player in players:
                        # Generate consistent player ID
                        player_id = self._generate_player_id(player["name"], player["previous_school"])
                        
                        # Add to new players dict
                        if player_id not in new_players:
                            new_players[player_id] = player
                        else:
                            # If player already exists from another source, merge information
                            existing = new_players[player_id]
                            existing["sources"] = existing.get("sources", [existing["source"]]) + [player["source"]]
                            
                            # Keep the most recent update
                            if player.get("last_updated") > existing.get("last_updated", ""):
                                existing.update(player)
                    
                    self.logger.info(f"Got {len(players)} players from {source} for {sport}")
                    
                except Exception as e:
                    self.logger.error(f"Error fetching data from {source} for {sport}: {str(e)}")
                    self.logger.error(traceback.format_exc())
        
        # Store previous data for comparison
        previous_players = self.players.copy()
        
        # Update current players
        self.players = new_players
        
        # Record the refresh
        self.last_refresh = datetime.now()
        
        # Save the data to memory
        self.remember(
            {
                "players": self.players,
                "last_refresh": self.last_refresh.isoformat(),
                "count": len(self.players)
            },
            "current_players"
        )
        
        # Log changes since last refresh
        if previous_players:
            new_ids = set(self.players.keys()) - set(previous_players.keys())
            removed_ids = set(previous_players.keys()) - set(self.players.keys())
            updated_ids = {
                pid for pid in set(self.players.keys()) & set(previous_players.keys())
                if self.players[pid] != previous_players[pid]
            }
            
            self.logger.info(f"Changes: {len(new_ids)} new, {len(removed_ids)} removed, {len(updated_ids)} updated")
            
            # Record changes
            if new_ids or removed_ids or updated_ids:
                changes = {
                    "timestamp": datetime.now().isoformat(),
                    "new_players": [self.players[pid] for pid in new_ids],
                    "removed_players": [previous_players[pid] for pid in removed_ids],
                    "updated_players": [
                        {"before": previous_players[pid], "after": self.players[pid]}
                        for pid in updated_ids
                    ]
                }
                
                # Remember changes with timestamp
                self.remember(changes, f"changes_{int(time.time())}", ["changes"])
                
                # Learn from these changes if learning is enabled
                if self.learning and (new_ids or updated_ids):
                    self._learn_from_changes(changes)
        
        return True
    
    def _generate_player_id(self, name: str, previous_school: str) -> str:
        """Generate a consistent player ID based on name and previous school"""
        return f"{name.lower().replace(' ', '_')}_{previous_school.lower().replace(' ', '_')}"
    
    def _learn_from_changes(self, changes: Dict[str, Any]) -> None:
        """Learn from changes in player data"""
        # Extract features and targets for commitment prediction model
        X_commitment = []
        y_commitment = []
        
        # Extract features and targets for rating prediction model
        X_rating = []
        y_rating = []
        
        # Process new and updated players
        for player in changes.get("new_players", []) + [p["after"] for p in changes.get("updated_players", [])]:
            # Features for commitment prediction
            # [rating, stars, offers, visits, days_in_portal]
            if "rating" in player and "offers" in player:
                days_in_portal = 0
                if "last_updated" in player:
                    try:
                        last_updated = datetime.fromisoformat(player["last_updated"])
                        days_in_portal = (datetime.now() - last_updated).days
                    except:
                        pass
                
                features = [
                    player.get("rating", 0),
                    player.get("stars", 0),
                    player.get("offers", 0),
                    player.get("visits", 0),
                    days_in_portal
                ]
                
                X_commitment.append(features)
                y_commitment.append(1 if player.get("committed", False) else 0)
                
                # Features for rating prediction
                # [stars, height, weight, offers]
                if "stars" in player and "height" in player:
                    rating_features = [
                        player.get("stars", 0),
                        player.get("height", 0),
                        player.get("weight", 0),
                        player.get("offers", 0)
                    ]
                    
                    X_rating.append(rating_features)
                    y_rating.append(player.get("rating", 0))
        
        # Train models if we have enough data
        if len(X_commitment) >= 5:
            self.logger.info(f"Training commitment predictor model with {len(X_commitment)} samples")
            self.learn("commitment_predictor", X_commitment, y_commitment)
        
        if len(X_rating) >= 5:
            self.logger.info(f"Training rating predictor model with {len(X_rating)} samples")
            self.learn("rating_predictor", X_rating, y_rating)
    
    def predict_commitment(self, player: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict if a player will commit soon and to which school
        
        Returns:
            Dictionary with prediction information
        """
        if not self.learning:
            return {"error": "Learning capabilities not available"}
        
        # Extract features for prediction
        days_in_portal = 0
        if "last_updated" in player:
            try:
                last_updated = datetime.fromisoformat(player["last_updated"])
                days_in_portal = (datetime.now() - last_updated).days
            except:
                pass
        
        features = [
            player.get("rating", 0),
            player.get("stars", 0),
            player.get("offers", 0),
            player.get("visits", 0),
            days_in_portal
        ]
        
        # Make prediction
        try:
            commitment_prob = self.predict("commitment_predictor", features)
            
            # Get prediction information
            result = {
                "player_name": player["name"],
                "commitment_probability": float(commitment_prob),
                "likely_to_commit_soon": commitment_prob > 0.7,
                "prediction_time": datetime.now().isoformat()
            }
            
            # Save prediction to memory
            self.remember(
                {
                    "player": player,
                    "prediction": result
                },
                f"prediction_{player['name'].replace(' ', '_')}_{int(time.time())}",
                ["prediction"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making commitment prediction: {str(e)}")
            return {"error": str(e)}
    
    def analyze_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in the transfer portal data
        
        Returns:
            Dictionary with trend analysis
        """
        if not self.players:
            return {"error": "No data available"}
        
        # Convert to DataFrame if pandas is available
        if PANDAS_AVAILABLE:
            try:
                df = pd.DataFrame(list(self.players.values()))
                
                # Get trends by position
                position_counts = df["position"].value_counts().to_dict()
                
                # Get trends by previous school
                school_counts = df["previous_school"].value_counts().to_dict()
                
                # Get commitment rate
                commitment_rate = df["committed"].mean()
                
                # Average rating by position
                avg_rating_by_position = df.groupby("position")["rating"].mean().to_dict()
                
                # Create trends report
                trends = {
                    "total_players": len(self.players),
                    "by_position": position_counts,
                    "by_school": school_counts,
                    "commitment_rate": commitment_rate,
                    "avg_rating_by_position": avg_rating_by_position,
                    "analysis_time": datetime.now().isoformat()
                }
                
                # Save trends to memory
                self.remember(trends, f"trends_{int(time.time())}", ["trends", "analysis"])
                
                return trends
                
            except Exception as e:
                self.logger.error(f"Error analyzing trends: {str(e)}")
                return {"error": str(e)}
        else:
            # Basic analysis without pandas
            positions = {}
            schools = {}
            committed_count = 0
            
            for player in self.players.values():
                # Count positions
                pos = player.get("position", "Unknown")
                positions[pos] = positions.get(pos, 0) + 1
                
                # Count schools
                school = player.get("previous_school", "Unknown")
                schools[school] = schools.get(school, 0) + 1
                
                # Count committed players
                if player.get("committed", False):
                    committed_count += 1
            
            # Calculate commitment rate
            commitment_rate = committed_count / len(self.players) if self.players else 0
            
            # Create trends report
            trends = {
                "total_players": len(self.players),
                "by_position": positions,
                "by_school": schools,
                "commitment_rate": commitment_rate,
                "analysis_time": datetime.now().isoformat()
            }
            
            # Save trends to memory
            self.remember(trends, f"trends_{int(time.time())}", ["trends", "analysis"])
            
            return trends
    
    def get_player_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Find a player by name (partial match)"""
        name_lower = name.lower()
        
        # Search for player by name
        for player_id, player in self.players.items():
            if name_lower in player.get("name", "").lower():
                return player
        
        return None
    
    def get_school_transfers(self, school: str) -> List[Dict[str, Any]]:
        """Get players transferring from a specific school"""
        school_lower = school.lower()
        
        # Find players from the school
        result = []
        for player in self.players.values():
            if school_lower in player.get("previous_school", "").lower():
                result.append(player)
        
        return result
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the agent's main task
        
        Args:
            **kwargs: Additional arguments
            
        Returns:
            Results of the agent's execution
        """
        self.logger.info(f"Running transfer portal agent with options: {kwargs}")
        
        # Determine if we need to refresh data
        force_refresh = kwargs.get("force_refresh", False)
        if force_refresh or not self.players or not self.last_refresh or (
            datetime.now() - self.last_refresh).total_seconds() > self.config["refresh_interval"]:
            self.refresh_data()
        
        # Generate analyses
        trends = self.analyze_trends()
        
        # Make predictions for uncommitted players
        predictions = []
        for player in self.players.values():
            if not player.get("committed", False):
                prediction = self.predict_commitment(player)
                predictions.append(prediction)
        
        # Sort predictions by commitment probability
        predictions.sort(key=lambda x: x.get("commitment_probability", 0), reverse=True)
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "player_count": len(self.players),
            "trends": trends,
            "top_predictions": predictions[:5],  # Top 5 predictions
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None
        }
        
        # Save report to memory
        self.remember(report, f"report_{int(time.time())}", ["report"])
        
        # Save models if we've learned anything
        if self.learning:
            self.save_models()
        
        self.logger.info(f"Transfer portal agent completed run")
        return report

# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transfer Portal Agent")
    parser.add_argument("--memory", type=str, default="sqlite", choices=["file", "sqlite", "duckdb", "vector", "mem0"], help="Memory type")
    parser.add_argument("--force-refresh", action="store_true", help="Force data refresh")
    parser.add_argument("--player", type=str, help="Search for a specific player")
    parser.add_argument("--school", type=str, help="Search for players from a specific school")
    
    args = parser.parse_args()
    
    # Create agent
    agent = TransferPortalAgent(memory_type=args.memory)
    
    # Handle specific commands
    if args.player:
        # Load existing data if available
        stored_data = agent.recall("current_players")
        if stored_data and "players" in stored_data:
            agent.players = stored_data["players"]
            agent.last_refresh = datetime.fromisoformat(stored_data["last_refresh"])
        else:
            # If no stored data, refresh
            agent.refresh_data()
        
        # Search for player
        player = agent.get_player_by_name(args.player)
        if player:
            print(json.dumps(player, indent=2))
            
            # Make prediction if player is not committed
            if not player.get("committed", False):
                prediction = agent.predict_commitment(player)
                print("\nCommitment Prediction:")
                print(json.dumps(prediction, indent=2))
        else:
            print(f"No player found matching '{args.player}'")
        
    elif args.school:
        # Load existing data if available
        stored_data = agent.recall("current_players")
        if stored_data and "players" in stored_data:
            agent.players = stored_data["players"]
            agent.last_refresh = datetime.fromisoformat(stored_data["last_refresh"])
        else:
            # If no stored data, refresh
            agent.refresh_data()
        
        # Get players from school
        players = agent.get_school_transfers(args.school)
        print(f"Found {len(players)} players transferring from '{args.school}':")
        for player in players:
            status = "Committed to " + player["commitment_school"] if player.get("committed") else "Uncommitted"
            print(f"- {player['name']} ({player['position']}): {status}")
        
    else:
        # Run the full agent process
        result = agent.run(force_refresh=args.force_refresh)
        
        # Print summary
        print("Transfer Portal Agent Report:")
        print(f"Time: {result['timestamp']}")
        print(f"Total Players: {result['player_count']}")
        print(f"Last Refresh: {result['last_refresh']}")
        
        print("\nTop Transfer Predictions:")
        for pred in result["top_predictions"]:
            prob = pred.get("commitment_probability", 0) * 100
            name = pred.get("player_name", "Unknown")
            print(f"- {name}: {prob:.1f}% chance of committing soon")
        
        print("\nPortal Trends:")
        trends = result["trends"]
        print(f"Overall Commitment Rate: {trends.get('commitment_rate', 0) * 100:.1f}%")
        
        # Top positions
        positions = trends.get("by_position", {})
        top_positions = sorted(positions.items(), key=lambda x: x[1], reverse=True)[:3]
        print("\nTop Positions in Portal:")
        for pos, count in top_positions:
            print(f"- {pos}: {count} players")
        
        # Top schools
        schools = trends.get("by_school", {})
        top_schools = sorted(schools.items(), key=lambda x: x[1], reverse=True)[:3]
        print("\nTop Schools Losing Players:")
        for school, count in top_schools:
            print(f"- {school}: {count} players") 