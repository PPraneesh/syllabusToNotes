# utils/manim_generator.py
import os
import subprocess
import re
import shutil
from pathlib import Path

class ManimGenerator:
    def __init__(self):
        # Define output directories
        self.manim_output_dir = "manim_output"
        self.video_output_dir = os.path.join(self.manim_output_dir, "videos")
        self.public_video_dir = os.path.join("notes_assets", "videos")
        
        # Ensure directories exist
        os.makedirs(self.manim_output_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)
        os.makedirs(self.public_video_dir, exist_ok=True)
    
    def ensure_environment(self):
        """Check if Manim is properly installed"""
        try:
            # Try running a basic Manim command
            result = subprocess.run(
                ["manim", "--help"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def create_animation(self, topic, concepts=None):
        """Generate Manim animation for a topic"""
        try:
            # Generate unique scene name (sanitized)
            scene_name = self._sanitize_scene_name(topic)
            
            # Generate Manim Python script
            script_content = self._generate_manim_script(scene_name, topic, concepts)
            script_path = os.path.join(self.manim_output_dir, f"{scene_name}.py")
            
            # Save script
            with open(script_path, "w") as f:
                f.write(script_content)
            
            # Run Manim to generate animation
            cmd = ["manim", "-pql", script_path, scene_name]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.returncode != 0:
                return False, f"Manim error: {result.stderr}"
            
            # Find the output video path from Manim's output
            video_path = self._find_video_path(scene_name, result.stdout)
            
            if not video_path or not os.path.exists(video_path):
                # Fallback: Search for the video using the expected pattern
                video_path = self._find_video_by_pattern(scene_name)
            
            if not video_path or not os.path.exists(video_path):
                return False, "Could not locate generated video"
            
            # Copy to public directory with a simplified name
            public_path = self._copy_to_public_dir(video_path, scene_name)
            
            return True, public_path
        
        except Exception as e:
            return False, f"Animation generation error: {str(e)}"
    
    def _sanitize_scene_name(self, topic):
        """Convert topic to a valid Python class name"""
        # Remove non-alphanumeric characters and replace spaces with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9 ]', '', topic)
        sanitized = re.sub(r'\s+', '_', sanitized)
        # Make sure it starts with a letter and capitalize first letter of each word
        sanitized = ''.join(word.capitalize() for word in sanitized.split('_'))
        # Ensure it's a valid class name
        if not sanitized or not sanitized[0].isalpha():
            sanitized = "Topic" + sanitized
        return sanitized
    
    def _generate_manim_script(self, scene_name, topic, concepts=None):
        """Generate Python code for Manim animation"""
        # Basic animation if no concepts provided
        if not concepts or len(concepts) == 0:
            return f"""from manim import *

class {scene_name}(Scene):
    def construct(self):
        title = Text("{topic}", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.scale(0.6).to_edge(UP))
        
        bullet = Text("• Key concept of {topic}", font_size=36)
        bullet.next_to(title, DOWN, buff=1)
        self.play(Write(bullet))
        self.wait(2)
"""
        
        # More complex animation with concepts
        concepts_str = '", "'.join(concepts[:3])  # Limit to 3 concepts for simplicity
        return f"""from manim import *

class {scene_name}(Scene):
    def construct(self):
        title = Text("{topic}", font_size=48)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.scale(0.6).to_edge(UP))
        
        concepts = ["{concepts_str}"]
        
        prev_bullet = None
        for i, concept in enumerate(concepts):
            bullet = Text(f"• {{concept}}", font_size=36)
            
            if prev_bullet:
                bullet.next_to(prev_bullet, DOWN, buff=0.5)
            else:
                bullet.next_to(title, DOWN, buff=1)
                
            self.play(Write(bullet))
            self.wait(1)
            prev_bullet = bullet
        
        self.wait(2)
"""
    
    def _find_video_path(self, scene_name, output_text):
        """Extract video path from Manim output"""
        # Try to find the path in the output text
        path_match = re.search(r'File ready at (.*\.mp4)', output_text)
        if path_match:
            return path_match.group(1)
        return None
    
    def _find_video_by_pattern(self, scene_name):
        """Find video by exploring directories using the expected pattern"""
        # Common Manim output patterns
        patterns = [
            # Pattern 1: Direct in videos folder
            os.path.join(self.video_output_dir, f"{scene_name}.mp4"),
            
            # Pattern 2: In scene name subfolder
            os.path.join(self.video_output_dir, scene_name, "480p15", f"{scene_name}.mp4"),
            os.path.join(self.video_output_dir, scene_name, "720p30", f"{scene_name}.mp4"),
            
            # Pattern 3: Using scene folder pattern from newer Manim versions
            os.path.join("media", "videos", scene_name, "480p15", f"{scene_name}.mp4"),
            os.path.join("media", "videos", scene_name, "720p30", f"{scene_name}.mp4"),
        ]
        
        # Try each pattern
        for pattern in patterns:
            if os.path.exists(pattern):
                return pattern
        
        # If no exact match, try to find by scene name in any subfolder
        for root, dirs, files in os.walk(os.path.dirname(self.video_output_dir)):
            for file in files:
                if file.endswith(".mp4") and scene_name in file:
                    return os.path.join(root, file)
        
        return None
    
    def _copy_to_public_dir(self, video_path, scene_name):
        """Copy video to public directory with simplified name"""
        # Generate a clean, simplified filename
        dest_filename = f"{scene_name}_animation.mp4"
        dest_path = os.path.join(self.public_video_dir, dest_filename)
        
        # Copy the file
        shutil.copy2(video_path, dest_path)
        
        return dest_path