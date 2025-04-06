#!/usr/bin/env python3
"""
Autonomous Scheduler for Single-File Agents
Purpose: Manages and schedules the execution of autonomous agents
Version: 1.0.0

Usage:
  python autonomous_scheduler.py add <agent_id> <agent_script> [--type interval|daily|cron] [--interval seconds] [--time HH:MM] [--cron expression] [--memory type] [--args "arg1 arg2"]
  python autonomous_scheduler.py remove <agent_id>
  python autonomous_scheduler.py list
  python autonomous_scheduler.py start <agent_id>
  python autonomous_scheduler.py stop <agent_id>
  python autonomous_scheduler.py status <agent_id>
  python autonomous_scheduler.py run-now <agent_id>
  python autonomous_scheduler.py run-all
  
Requirements:
  - Python 3.8+
  - pip install schedule apscheduler
"""

import os
import sys
import json
import time
import logging
import argparse
import importlib.util
import subprocess
import signal
import datetime
import traceback
from typing import Dict, List, Any, Optional
from pathlib import Path
import multiprocessing
from multiprocessing import Process, Manager

# Try importing scheduling libraries
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.interval import IntervalTrigger
    from apscheduler.triggers.date import DateTrigger
    APSCHEDULER_AVAILABLE = True
except ImportError:
    APSCHEDULER_AVAILABLE = False

# Configure logging
os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/scheduler.log"), mode='a')
    ]
)
logger = logging.getLogger("scheduler")

# Constants
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")
CONFIG_FILE = os.path.join(CONFIG_DIR, "scheduler.json")
AGENTS_DIR = os.path.dirname(os.path.dirname(__file__))

# Ensure config directory exists
os.makedirs(CONFIG_DIR, exist_ok=True)

class AgentProcess:
    """Represents a running agent process"""
    
    def __init__(self, agent_id: str, proc: Optional[multiprocessing.Process] = None):
        self.agent_id = agent_id
        self.proc = proc
        self.start_time = datetime.datetime.now() if proc else None
        self.status = "running" if proc and proc.is_alive() else "stopped"
    
    def is_running(self) -> bool:
        """Check if the agent process is running"""
        return self.proc is not None and self.proc.is_alive()
    
    def stop(self) -> bool:
        """Stop the agent process"""
        if self.is_running():
            self.proc.terminate()
            self.proc.join(timeout=5)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(timeout=2)
            
            self.status = "stopped"
            return True
        
        return False
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the process"""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "pid": self.proc.pid if self.proc else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "running_time": str(datetime.datetime.now() - self.start_time) if self.start_time else None
        }

class SchedulerConfig:
    """Configuration for the autonomous scheduler"""
    
    def __init__(self, config_file: str = CONFIG_FILE):
        self.config_file = config_file
        self.agents = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.agents = data.get("agents", {})
            except Exception as e:
                logger.error(f"Error loading config: {e}")
    
    def save(self) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump({"agents": self.agents}, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def add_agent(self, agent_id: str, agent_config: Dict[str, Any]) -> None:
        """Add or update an agent configuration"""
        self.agents[agent_id] = agent_config
        self.save()
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent configuration"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.save()
            return True
        return False
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent configuration"""
        return self.agents.get(agent_id)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all agent configurations"""
        return [
            {"agent_id": agent_id, **config}
            for agent_id, config in self.agents.items()
        ]

class AutonomousScheduler:
    """Manages scheduled execution of autonomous agents"""
    
    def __init__(self):
        """Initialize the scheduler"""
        self.config = SchedulerConfig()
        self.processes = {}  # agent_id -> AgentProcess
        
        # Initialize scheduler
        if APSCHEDULER_AVAILABLE:
            self.scheduler = BackgroundScheduler()
            self.scheduler.start()
        elif SCHEDULE_AVAILABLE:
            logger.warning("APScheduler not available, falling back to schedule library (less functionality)")
        else:
            logger.warning("No scheduling library available. Install with: pip install apscheduler schedule")
            self.scheduler = None
    
    def add_agent(
        self,
        agent_id: str,
        script_path: str,
        schedule_type: str = "interval",
        interval: int = 3600,
        time_str: str = None,
        cron_expr: str = None,
        memory_type: str = "sqlite",
        args: List[str] = None
    ) -> bool:
        """Add an agent to the scheduler"""
        # Validate script path
        if not os.path.exists(script_path):
            logger.error(f"Script {script_path} does not exist")
            return False
        
        # Validate schedule type
        if schedule_type not in ["interval", "daily", "cron"]:
            logger.error(f"Invalid schedule type: {schedule_type}")
            return False
        
        # Prepare agent configuration
        agent_config = {
            "script_path": script_path,
            "schedule_type": schedule_type,
            "interval": interval,
            "time": time_str,
            "cron": cron_expr,
            "memory_type": memory_type,
            "args": args or [],
            "enabled": True,
            "last_run": None,
            "next_run": None
        }
        
        # Save to configuration
        self.config.add_agent(agent_id, agent_config)
        
        # Schedule the agent
        self._schedule_agent(agent_id, agent_config)
        
        logger.info(f"Added agent {agent_id} with {schedule_type} schedule")
        return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the scheduler"""
        # Stop if running
        self.stop_agent(agent_id)
        
        # Remove from scheduler
        if APSCHEDULER_AVAILABLE:
            try:
                self.scheduler.remove_job(agent_id)
            except:
                pass
        
        # Remove from configuration
        return self.config.remove_agent(agent_id)
    
    def _schedule_agent(self, agent_id: str, agent_config: Dict[str, Any]) -> None:
        """Schedule an agent based on its configuration"""
        if not agent_config.get("enabled", True):
            logger.info(f"Agent {agent_id} is disabled, not scheduling")
            return
        
        schedule_type = agent_config.get("schedule_type", "interval")
        
        if APSCHEDULER_AVAILABLE:
            # Remove existing job if any
            try:
                self.scheduler.remove_job(agent_id)
            except:
                pass
            
            # Create trigger based on schedule type
            if schedule_type == "interval":
                interval = agent_config.get("interval", 3600)  # Default 1 hour
                trigger = IntervalTrigger(seconds=interval)
                next_run = datetime.datetime.now() + datetime.timedelta(seconds=interval)
            
            elif schedule_type == "daily":
                time_str = agent_config.get("time", "00:00")
                hour, minute = map(int, time_str.split(":"))
                trigger = CronTrigger(hour=hour, minute=minute)
                
                # Calculate next run time
                now = datetime.datetime.now()
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run < now:
                    next_run += datetime.timedelta(days=1)
            
            elif schedule_type == "cron":
                cron_expr = agent_config.get("cron", "0 0 * * *")
                trigger = CronTrigger.from_crontab(cron_expr)
                
                # Calculate next run time
                next_run = trigger.get_next_fire_time(None, datetime.datetime.now())
            
            else:
                logger.error(f"Unknown schedule type: {schedule_type}")
                return
            
            # Schedule the job
            self.scheduler.add_job(
                self._run_agent,
                trigger=trigger,
                args=[agent_id],
                id=agent_id,
                replace_existing=True
            )
            
            # Update next run time
            agent_config["next_run"] = next_run.isoformat()
            self.config.save()
            
            logger.info(f"Scheduled agent {agent_id} to run next at {next_run.isoformat()}")
        
        elif SCHEDULE_AVAILABLE:
            # This is a simplified version using the schedule library
            # It only supports interval and daily schedules
            
            # Clear existing schedules for this agent
            schedule.clear(agent_id)
            
            if schedule_type == "interval":
                interval = agent_config.get("interval", 3600)  # Default 1 hour
                schedule.every(interval).seconds.do(
                    lambda: self._run_agent(agent_id)
                ).tag(agent_id)
                
                next_run = datetime.datetime.now() + datetime.timedelta(seconds=interval)
            
            elif schedule_type == "daily":
                time_str = agent_config.get("time", "00:00")
                schedule.every().day.at(time_str).do(
                    lambda: self._run_agent(agent_id)
                ).tag(agent_id)
                
                # Calculate next run time
                hour, minute = map(int, time_str.split(":"))
                now = datetime.datetime.now()
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run < now:
                    next_run += datetime.timedelta(days=1)
            
            else:
                logger.error(f"Schedule type {schedule_type} not supported with schedule library")
                return
            
            # Update next run time
            agent_config["next_run"] = next_run.isoformat()
            self.config.save()
            
            logger.info(f"Scheduled agent {agent_id} with schedule library to run next at {next_run.isoformat()}")
        
        else:
            logger.error("No scheduling library available")
    
    def _run_agent(self, agent_id: str) -> None:
        """Run an agent script in a separate process"""
        agent_config = self.config.get_agent(agent_id)
        if not agent_config:
            logger.error(f"Agent {agent_id} not found")
            return
        
        # Update last run time
        agent_config["last_run"] = datetime.datetime.now().isoformat()
        
        # Get script path and args
        script_path = agent_config.get("script_path")
        memory_type = agent_config.get("memory_type", "sqlite")
        args = agent_config.get("args", [])
        
        # Prepare command arguments
        cmd_args = [sys.executable, script_path, "--memory", memory_type]
        if args:
            cmd_args.extend(args)
        
        logger.info(f"Running agent {agent_id} with command: {' '.join(cmd_args)}")
        
        # Create and start process
        def run_process():
            try:
                subprocess.run(cmd_args, check=True)
                logger.info(f"Agent {agent_id} completed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"Agent {agent_id} failed with exit code {e.returncode}")
            except Exception as e:
                logger.error(f"Error running agent {agent_id}: {e}")
                logger.error(traceback.format_exc())
        
        # Run in a separate process
        proc = Process(target=run_process)
        proc.daemon = True
        proc.start()
        
        # Store process
        self.processes[agent_id] = AgentProcess(agent_id, proc)
        
        # Update next run time for interval schedules
        if agent_config.get("schedule_type") == "interval":
            interval = agent_config.get("interval", 3600)
            next_run = datetime.datetime.now() + datetime.timedelta(seconds=interval)
            agent_config["next_run"] = next_run.isoformat()
        
        # Save updated config
        self.config.save()
    
    def run_agent_sync(self, agent_id: str) -> bool:
        """Run an agent synchronously (blocking)"""
        agent_config = self.config.get_agent(agent_id)
        if not agent_config:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        # Get script path and args
        script_path = agent_config.get("script_path")
        memory_type = agent_config.get("memory_type", "sqlite")
        args = agent_config.get("args", [])
        
        # Prepare command arguments
        cmd_args = [sys.executable, script_path, "--memory", memory_type]
        if args:
            cmd_args.extend(args)
        
        logger.info(f"Running agent {agent_id} synchronously with command: {' '.join(cmd_args)}")
        
        try:
            # Update last run time
            agent_config["last_run"] = datetime.datetime.now().isoformat()
            self.config.save()
            
            # Run synchronously
            subprocess.run(cmd_args, check=True)
            
            logger.info(f"Agent {agent_id} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Agent {agent_id} failed with exit code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"Error running agent {agent_id}: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def start_agent(self, agent_id: str) -> bool:
        """Enable and schedule an agent"""
        agent_config = self.config.get_agent(agent_id)
        if not agent_config:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        # Enable the agent
        agent_config["enabled"] = True
        self.config.save()
        
        # Schedule the agent
        self._schedule_agent(agent_id, agent_config)
        
        logger.info(f"Started agent {agent_id}")
        return True
    
    def stop_agent(self, agent_id: str) -> bool:
        """Disable and unschedule an agent"""
        agent_config = self.config.get_agent(agent_id)
        if not agent_config:
            logger.error(f"Agent {agent_id} not found")
            return False
        
        # Disable the agent
        agent_config["enabled"] = False
        self.config.save()
        
        # Remove from scheduler
        if APSCHEDULER_AVAILABLE:
            try:
                self.scheduler.remove_job(agent_id)
            except:
                pass
        
        if SCHEDULE_AVAILABLE:
            schedule.clear(agent_id)
        
        # Stop any running process
        if agent_id in self.processes:
            self.processes[agent_id].stop()
            del self.processes[agent_id]
        
        logger.info(f"Stopped agent {agent_id}")
        return True
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed status of an agent"""
        agent_config = self.config.get_agent(agent_id)
        if not agent_config:
            logger.error(f"Agent {agent_id} not found")
            return {"error": f"Agent {agent_id} not found"}
        
        # Check if process is running
        is_running = agent_id in self.processes and self.processes[agent_id].is_running()
        process_info = self.processes[agent_id].get_info() if agent_id in self.processes else {}
        
        # Build status
        status = {
            "agent_id": agent_id,
            "script_path": agent_config.get("script_path"),
            "schedule_type": agent_config.get("schedule_type"),
            "enabled": agent_config.get("enabled", True),
            "last_run": agent_config.get("last_run"),
            "next_run": agent_config.get("next_run"),
            "is_running": is_running,
            "process": process_info
        }
        
        if agent_config.get("schedule_type") == "interval":
            status["interval"] = agent_config.get("interval")
        elif agent_config.get("schedule_type") == "daily":
            status["time"] = agent_config.get("time")
        elif agent_config.get("schedule_type") == "cron":
            status["cron"] = agent_config.get("cron")
        
        return status
    
    def list_agents_status(self) -> List[Dict[str, Any]]:
        """List all agents with their status"""
        agent_statuses = []
        
        for agent_id, config in self.config.agents.items():
            is_running = agent_id in self.processes and self.processes[agent_id].is_running()
            
            status = {
                "agent_id": agent_id,
                "script_path": config.get("script_path"),
                "enabled": config.get("enabled", True),
                "is_running": is_running,
                "last_run": config.get("last_run"),
                "next_run": config.get("next_run"),
                "schedule_type": config.get("schedule_type")
            }
            
            agent_statuses.append(status)
        
        return agent_statuses
    
    def run_all_agents(self) -> Dict[str, bool]:
        """Run all enabled agents immediately"""
        results = {}
        
        for agent_id, config in self.config.agents.items():
            if config.get("enabled", True):
                logger.info(f"Running agent {agent_id}")
                success = self.run_agent_sync(agent_id)
                results[agent_id] = success
        
        return results
    
    def run(self) -> None:
        """Run the scheduler main loop (using schedule library)"""
        if not SCHEDULE_AVAILABLE:
            logger.error("Schedule library not available. Cannot run scheduler loop.")
            return
        
        logger.info("Starting scheduler loop with schedule library")
        
        # Schedule all agents
        for agent_id, config in self.config.agents.items():
            if config.get("enabled", True):
                self._schedule_agent(agent_id, config)
        
        # Main loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        except Exception as e:
            logger.error(f"Error in scheduler loop: {e}")
            logger.error(traceback.format_exc())

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Autonomous Agent Scheduler")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add agent
    add_parser = subparsers.add_parser("add", help="Add an agent to the scheduler")
    add_parser.add_argument("agent_id", help="ID for the agent")
    add_parser.add_argument("script_path", help="Path to the agent script")
    add_parser.add_argument("--type", choices=["interval", "daily", "cron"], default="interval", help="Schedule type")
    add_parser.add_argument("--interval", type=int, default=3600, help="Interval in seconds (for interval schedule)")
    add_parser.add_argument("--time", help="Time of day in HH:MM format (for daily schedule)")
    add_parser.add_argument("--cron", help="Cron expression (for cron schedule)")
    add_parser.add_argument("--memory", default="sqlite", help="Memory type for agent")
    add_parser.add_argument("--args", help="Additional arguments for the agent script")
    
    # Remove agent
    remove_parser = subparsers.add_parser("remove", help="Remove an agent from the scheduler")
    remove_parser.add_argument("agent_id", help="ID of the agent to remove")
    
    # List agents
    subparsers.add_parser("list", help="List all scheduled agents")
    
    # Start agent
    start_parser = subparsers.add_parser("start", help="Start (enable) an agent")
    start_parser.add_argument("agent_id", help="ID of the agent to start")
    
    # Stop agent
    stop_parser = subparsers.add_parser("stop", help="Stop (disable) an agent")
    stop_parser.add_argument("agent_id", help="ID of the agent to stop")
    
    # Agent status
    status_parser = subparsers.add_parser("status", help="Get status of an agent")
    status_parser.add_argument("agent_id", help="ID of the agent to check")
    
    # Run agent now
    run_parser = subparsers.add_parser("run-now", help="Run an agent immediately")
    run_parser.add_argument("agent_id", help="ID of the agent to run")
    
    # Run all agents
    subparsers.add_parser("run-all", help="Run all enabled agents immediately")
    
    # Run scheduler
    subparsers.add_parser("run", help="Run scheduler loop (only needed with schedule library)")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    scheduler = AutonomousScheduler()
    
    if args.command == "add":
        # Parse additional args if provided
        additional_args = args.args.split() if args.args else []
        
        success = scheduler.add_agent(
            args.agent_id,
            args.script_path,
            args.type,
            args.interval,
            args.time,
            args.cron,
            args.memory,
            additional_args
        )
        
        if success:
            print(f"Added agent {args.agent_id} to scheduler")
        else:
            print(f"Failed to add agent {args.agent_id}")
    
    elif args.command == "remove":
        success = scheduler.remove_agent(args.agent_id)
        if success:
            print(f"Removed agent {args.agent_id} from scheduler")
        else:
            print(f"Failed to remove agent {args.agent_id}")
    
    elif args.command == "list":
        statuses = scheduler.list_agents_status()
        if statuses:
            print("Scheduled agents:")
            for status in statuses:
                enabled = "ENABLED" if status.get("enabled", True) else "DISABLED"
                running = "RUNNING" if status.get("is_running", False) else "STOPPED"
                last_run = status.get("last_run", "Never")
                next_run = status.get("next_run", "Not scheduled")
                schedule_type = status.get("schedule_type", "unknown")
                
                print(f"- {status['agent_id']} ({enabled}, {running})")
                print(f"  Script: {status['script_path']}")
                print(f"  Schedule: {schedule_type}")
                print(f"  Last run: {last_run}")
                print(f"  Next run: {next_run}")
                print()
        else:
            print("No agents scheduled")
    
    elif args.command == "start":
        success = scheduler.start_agent(args.agent_id)
        if success:
            print(f"Started agent {args.agent_id}")
        else:
            print(f"Failed to start agent {args.agent_id}")
    
    elif args.command == "stop":
        success = scheduler.stop_agent(args.agent_id)
        if success:
            print(f"Stopped agent {args.agent_id}")
        else:
            print(f"Failed to stop agent {args.agent_id}")
    
    elif args.command == "status":
        status = scheduler.get_agent_status(args.agent_id)
        if "error" in status:
            print(status["error"])
        else:
            enabled = "ENABLED" if status.get("enabled", True) else "DISABLED"
            running = "RUNNING" if status.get("is_running", False) else "STOPPED"
            
            print(f"Agent {args.agent_id} ({enabled}, {running})")
            print(f"Script: {status['script_path']}")
            print(f"Schedule type: {status['schedule_type']}")
            
            if status["schedule_type"] == "interval":
                print(f"Interval: {status.get('interval', 0)} seconds")
            elif status["schedule_type"] == "daily":
                print(f"Time: {status.get('time', '00:00')}")
            elif status["schedule_type"] == "cron":
                print(f"Cron: {status.get('cron', '* * * * *')}")
            
            print(f"Last run: {status.get('last_run', 'Never')}")
            print(f"Next run: {status.get('next_run', 'Not scheduled')}")
            
            if status.get("is_running", False):
                process = status.get("process", {})
                print("\nProcess information:")
                print(f"PID: {process.get('pid')}")
                print(f"Start time: {process.get('start_time')}")
                print(f"Running time: {process.get('running_time')}")
    
    elif args.command == "run-now":
        print(f"Running agent {args.agent_id} now...")
        success = scheduler.run_agent_sync(args.agent_id)
        if success:
            print(f"Agent {args.agent_id} completed successfully")
        else:
            print(f"Agent {args.agent_id} failed to run")
    
    elif args.command == "run-all":
        print("Running all enabled agents...")
        results = scheduler.run_all_agents()
        
        # Print results
        for agent_id, success in results.items():
            result = "completed successfully" if success else "failed"
            print(f"Agent {agent_id} {result}")
    
    elif args.command == "run":
        if APSCHEDULER_AVAILABLE:
            print("Using APScheduler - no need to run in foreground")
            print("All agents will run according to their schedules")
            print("Press Ctrl+C to exit")
            
            try:
                # Just keep the main thread alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Scheduler stopped")
        elif SCHEDULE_AVAILABLE:
            print("Starting scheduler loop with schedule library")
            print("Press Ctrl+C to exit")
            scheduler.run()
        else:
            print("No scheduling library available. Install with: pip install apscheduler schedule")
    
    else:
        print("No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main() 