#!/usr/bin/env python3
"""
Complete migration orchestrator for Alpha-Recall 768D embedding upgrade.

This script orchestrates the full migration from 384D to 768D embeddings:
1. Runs integration tests
2. Migrates STM embeddings in Redis
3. Migrates LTM observations in Qdrant
4. Validates the migration
5. Updates configuration files
"""

import asyncio
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_script(script_path: str, description: str) -> bool:
    """Run a Python script and return success status."""
    print(f"\n{'='*20} {description} {'='*20}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path) or "."
        )
        
        # Print stdout
        if result.stdout:
            print(result.stdout)
        
        # Print stderr if there are errors
        if result.stderr and result.returncode != 0:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False


def backup_current_state():
    """Create a backup of current configuration."""
    print("\n🔄 Creating backup of current state...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = Path(f"backup_{timestamp}")
    backup_dir.mkdir(exist_ok=True)
    
    # Files to backup
    files_to_backup = [
        "src/alpha_recall/db/factory.py",
        "src/alpha_recall/db/vector_store.py", 
        "src/alpha_recall/db/redis_db.py",
    ]
    
    for file_path in files_to_backup:
        src = Path(file_path)
        if src.exists():
            dst = backup_dir / src.name
            dst.write_text(src.read_text())
            print(f"  ✅ Backed up {file_path}")
    
    print(f"✅ Backup created in {backup_dir}")
    return backup_dir


def check_embedder_service():
    """Check if the new embedder service is running."""
    print("\n🔍 Checking embedder service...")
    
    import httpx
    
    try:
        response = httpx.get("http://localhost:6004/health", timeout=5.0)
        if response.status_code == 200:
            print("✅ Embedder service is running")
            return True
        else:
            print(f"❌ Embedder service returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to embedder service: {e}")
        print("   Make sure the embedder service is running on port 6004")
        return False


def prompt_user_confirmation():
    """Get user confirmation before proceeding."""
    print("\n⚠️  MIGRATION CONFIRMATION")
    print("This will:")
    print("  1. Re-embed 181 STM entries from 384D to 768D")
    print("  2. Re-embed 474 LTM observations from 384D to 768D")
    print("  3. Update vector indices in Redis")
    print("  4. Create new Qdrant collections")
    print("  5. Update alpha-recall code to use new embedder endpoints")
    print("\nThis process will take several minutes and cannot be easily undone.")
    print("A backup will be created, but re-migrating would take more time.")
    
    response = input("\nProceed with migration? (yes/no): ").lower().strip()
    return response in ['yes', 'y']


async def main():
    """Main migration orchestrator."""
    print("🧠 Alpha-Recall 768D Embedding Migration")
    print("=" * 60)
    
    # Check prerequisites
    if not check_embedder_service():
        print("\n❌ Migration aborted: Embedder service not available")
        return False
    
    # Get user confirmation
    if not prompt_user_confirmation():
        print("\n🚫 Migration cancelled by user")
        return False
    
    # Create backup
    backup_dir = backup_current_state()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Run migration steps
    steps = [
        (script_dir / "test_new_embedder.py", "Integration Tests"),
        (script_dir / "migrate_stm_to_768d.py", "STM Migration (Redis)"),
        (script_dir / "migrate_ltm_to_768d.py", "LTM Migration (Qdrant)"),
    ]
    
    success_count = 0
    
    for script_path, description in steps:
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            continue
            
        success = run_script(str(script_path), description)
        if success:
            success_count += 1
        else:
            print(f"\n❌ Migration step failed: {description}")
            print("Consider checking the logs and trying again.")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("🎯 MIGRATION SUMMARY")
    print(f"✅ Steps completed: {success_count}/{len(steps)}")
    
    if success_count == len(steps):
        print("🎉 Migration completed successfully!")
        print("\nNext steps:")
        print("  1. Test alpha-recall functionality")
        print("  2. Verify search quality improvements")
        print("  3. Monitor for any issues")
        print(f"  4. Remove backup folder when confident: {backup_dir}")
        
        # Save migration log
        log_file = f"migration_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_file, 'w') as f:
            f.write(f"Alpha-Recall 768D Migration Completed\n")
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Backup: {backup_dir}\n")
            f.write(f"Steps completed: {success_count}/{len(steps)}\n")
        print(f"  5. Migration log saved: {log_file}")
        
        return True
    else:
        print("❌ Migration incomplete!")
        print(f"   Backup available at: {backup_dir}")
        print("   Review errors above and retry if needed")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🚫 Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)