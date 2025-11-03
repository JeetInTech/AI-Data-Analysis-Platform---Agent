import pygame
import pygame.freetype
import math
import time
import threading
import random
import time
import queue
from pathlib import Path
from datetime import datetime
import json

# Try importing dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class PygameInterface:
    """Gamified Pygame interface for AI Data Analytics Platform."""
    
    def __init__(self, main_app):
        """Initialize Pygame interface with reference to main application."""
        self.main_app = main_app
        self.running = False
        
        # Initialize Pygame
        pygame.init()
        pygame.freetype.init()
        
        # Screen settings
        self.SCREEN_WIDTH = 1600
        self.SCREEN_HEIGHT = 1000
        self.FPS = 60
        
        # Colors (cyberpunk/gaming theme)
        self.colors = {
            'bg_primary': (15, 23, 42),      # Dark blue-gray
            'bg_secondary': (30, 41, 59),     # Lighter blue-gray
            'accent_primary': (59, 130, 246), # Bright blue
            'accent_secondary': (147, 51, 234), # Purple
            'success': (34, 197, 94),         # Green
            'warning': (249, 115, 22),        # Orange
            'error': (239, 68, 68),           # Red
            'text_primary': (248, 250, 252),  # Almost white
            'text_secondary': (148, 163, 184), # Gray
            'glow': (0, 255, 255),            # Cyan glow
            'achievement': (255, 215, 0),      # Gold
        }
        
        # Game state
        self.current_screen = "main_menu"
        self.transition_alpha = 0
        self.transitioning = False
        
        # UI Elements
        self.buttons = {}
        self.panels = {}
        self.animations = []
        self.particles = []
        
        # Gaming elements
        self.experience_points = 0
        self.level = 1
        self.achievements_unlocked = []
        self.current_quest = None
        
        # Data visualization elements
        self.data_nodes = []
        self.connections = []
        self.visualization_mode = "network"
        
        # Progress tracking
        self.progress_bars = {}
        self.status_messages = []
        
        # Sound effects (placeholder)
        self.sounds = {}
        
        # Communication with main app
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
    def initialize_display(self):
        """Initialize the game display."""
        try:
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("AI Analytics Professional Platform")
            
            # Load fonts with better hierarchy
            self.fonts = {
                'title': pygame.freetype.Font(None, 52),
                'heading': pygame.freetype.Font(None, 36),
                'subheading': pygame.freetype.Font(None, 28),
                'body': pygame.freetype.Font(None, 22),
                'small': pygame.freetype.Font(None, 18),
                'tiny': pygame.freetype.Font(None, 14)
            }
            
            # Create clock
            self.clock = pygame.time.Clock()
            
            # Initialize UI elements
            self.initialize_ui_elements()
            
            return True
            
        except Exception as e:
            print(f"Error initializing Pygame display: {e}")
            return False
    
    def initialize_ui_elements(self):
        """Initialize all UI elements and layouts."""
        # Main menu buttons with professional styling
        self.buttons['main_menu'] = [
            Button(self.SCREEN_WIDTH//2 - 175, 300, 350, 70, "START NEW ANALYSIS", 
                   self.colors['accent_primary'], self.start_new_analysis),
            Button(self.SCREEN_WIDTH//2 - 175, 390, 350, 70, "LOAD SESSION", 
                   self.colors['accent_secondary'], self.load_session),
            Button(self.SCREEN_WIDTH//2 - 175, 480, 350, 70, "VIEW ACHIEVEMENTS", 
                   self.colors['achievement'], self.show_achievements),
            Button(self.SCREEN_WIDTH//2 - 175, 570, 350, 70, "CONFIGURE SETTINGS", 
                   self.colors['text_secondary'], self.show_settings),
            Button(self.SCREEN_WIDTH//2 - 175, 660, 350, 70, "EXIT APPLICATION", 
                   self.colors['error'], self.exit_to_desktop)
        ]
        
        # Professional data analysis interface
        self.buttons['data_analysis'] = [
            Button(50, 100, 220, 55, "LOAD DATASET", self.colors['accent_primary'], self.load_data_dialog),
            Button(50, 170, 220, 55, "RUN AI ANALYSIS", self.colors['success'], self.run_ai_analysis),
            Button(50, 240, 220, 55, "CLEAN DATA", self.colors['warning'], self.clean_data),
            Button(50, 310, 220, 55, "TRAIN MODELS", self.colors['accent_secondary'], self.train_models),
            Button(50, 380, 220, 55, "CREATE VISUALIZATIONS", self.colors['achievement'], self.create_visualizations),
            Button(50, 450, 220, 55, "EXPLAIN MODELS", self.colors['glow'], self.explain_models),
            Button(50, 520, 220, 55, "EXPORT RESULTS", self.colors['info'], self.export_results),
            Button(50, 860, 120, 45, "MAIN MENU", self.colors['text_secondary'], self.return_to_menu)
        ]
        
        # Progress panels
        self.panels['progress'] = ProgressPanel(300, 50, 1000, 100)
        self.panels['data_preview'] = DataPreviewPanel(300, 170, 1000, 300)
        self.panels['model_performance'] = ModelPerformancePanel(300, 490, 1000, 200)
        self.panels['achievements'] = AchievementPanel(300, 710, 1000, 140)
        
        # Initialize particle system for effects
        self.initialize_particle_system()
    
    def initialize_particle_system(self):
        """Initialize particle system for visual effects."""
        self.particle_systems = {
            'data_flow': [],
            'success_burst': [],
            'level_up': [],
            'achievement_glow': []
        }
    
    def run(self):
        """Main game loop."""
        if not self.initialize_display():
            print("Failed to initialize Pygame display")
            return
        
        self.running = True
        
        try:
            while self.running:
                dt = self.clock.tick(self.FPS) / 1000.0  # Delta time in seconds
                
                # Handle events
                self.handle_events()
                
                # Update game state
                self.update(dt)
                
                # Render everything
                self.render()
                
                # Update display
                pygame.display.flip()
                
        except Exception as e:
            print(f"Error in main game loop: {e}")
        finally:
            self.cleanup()
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event)
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.handle_mouse_click(event)
            
            elif event.type == pygame.MOUSEMOTION:
                self.handle_mouse_motion(event)
    
    def handle_keydown(self, event):
        """Handle keyboard input."""
        if event.key == pygame.K_ESCAPE:
            if self.current_screen != "main_menu":
                self.transition_to_screen("main_menu")
            else:
                self.running = False
        
        elif event.key == pygame.K_F11:
            # Toggle fullscreen
            pygame.display.toggle_fullscreen()
        
        elif event.key == pygame.K_F1:
            # Show help
            self.show_help_overlay()
    
    def handle_mouse_click(self, event):
        """Handle mouse clicks."""
        mouse_pos = pygame.mouse.get_pos()
        
        # Check button clicks for current screen
        if self.current_screen in self.buttons:
            for button in self.buttons[self.current_screen]:
                if button.is_clicked(mouse_pos):
                    button.execute()
                    self.create_click_effect(mouse_pos)
        
        # Handle data visualization interactions
        if self.current_screen == "data_analysis":
            self.handle_data_visualization_click(mouse_pos)
    
    def handle_mouse_motion(self, event):
        """Handle mouse movement for hover effects."""
        mouse_pos = pygame.mouse.get_pos()
        
        # Update button hover states
        if self.current_screen in self.buttons:
            for button in self.buttons[self.current_screen]:
                button.update_hover(mouse_pos)
    
    def update(self, dt):
        """Update game state."""
        # Update transitions
        if self.transitioning:
            self.update_transition(dt)
        
        # Update particles
        self.update_particles(dt)
        
        # Update animations
        self.update_animations(dt)
        
        # Update progress bars
        self.update_progress_bars(dt)
        
        # Process commands from main app
        self.process_main_app_commands()
        
        # Update data visualization
        if self.current_screen == "data_analysis":
            self.update_data_visualization(dt)
    
    def render(self):
        """Render all game elements."""
        # Clear screen with gradient background
        self.render_background()
        
        # Render current screen
        if self.current_screen == "main_menu":
            self.render_main_menu()
        elif self.current_screen == "data_analysis":
            self.render_data_analysis()
        elif self.current_screen == "achievements":
            self.render_achievements()
        elif self.current_screen == "settings":
            self.render_settings()
        
        # Render particles and effects
        self.render_particles()
        
        # Render UI overlays
        self.render_ui_overlays()
        
        # Render transition effects
        if self.transitioning:
            self.render_transition()
    
    def render_background(self):
        """Render animated background."""
        # Create gradient background
        for y in range(self.SCREEN_HEIGHT):
            color_ratio = y / self.SCREEN_HEIGHT
            r = int(self.colors['bg_primary'][0] * (1 - color_ratio) + self.colors['bg_secondary'][0] * color_ratio)
            g = int(self.colors['bg_primary'][1] * (1 - color_ratio) + self.colors['bg_secondary'][1] * color_ratio)
            b = int(self.colors['bg_primary'][2] * (1 - color_ratio) + self.colors['bg_secondary'][2] * color_ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (self.SCREEN_WIDTH, y))
        
        # Add animated grid pattern
        self.render_grid_pattern()
        
        # Add floating data nodes
        self.render_floating_nodes()
    
    def render_grid_pattern(self):
        """Render animated grid pattern."""
        grid_size = 50
        time_offset = time.time() * 0.5
        
        for x in range(0, self.SCREEN_WIDTH, grid_size):
            for y in range(0, self.SCREEN_HEIGHT, grid_size):
                # Calculate glow intensity based on position and time
                glow_intensity = (math.sin(x * 0.01 + time_offset) + math.sin(y * 0.01 + time_offset)) * 0.5 + 0.5
                alpha = int(20 * glow_intensity)
                
                if alpha > 5:
                    color = (*self.colors['glow'], alpha)
                    glow_surface = pygame.Surface((2, 2), pygame.SRCALPHA)
                    glow_surface.fill(color)
                    self.screen.blit(glow_surface, (x, y))
    
    def render_floating_nodes(self):
        """Render floating data visualization nodes."""
        time_offset = time.time()
        
        for i in range(20):  # 20 floating nodes
            x = (math.sin(time_offset * 0.3 + i * 0.5) * 100) + self.SCREEN_WIDTH // 2
            y = (math.cos(time_offset * 0.2 + i * 0.3) * 50) + self.SCREEN_HEIGHT // 2
            
            # Node size varies with time
            size = 3 + math.sin(time_offset * 2 + i) * 2
            
            # Node color cycles
            color_intensity = (math.sin(time_offset + i * 0.7) + 1) * 0.5
            color = (
                int(self.colors['accent_primary'][0] * color_intensity),
                int(self.colors['accent_primary'][1] * color_intensity),
                int(self.colors['accent_primary'][2] * color_intensity)
            )
            
            pygame.draw.circle(self.screen, color, (int(x), int(y)), int(size))
            
            # Draw connections between nearby nodes
            for j in range(i + 1, min(i + 3, 20)):
                x2 = (math.sin(time_offset * 0.3 + j * 0.5) * 100) + self.SCREEN_WIDTH // 2
                y2 = (math.cos(time_offset * 0.2 + j * 0.3) * 50) + self.SCREEN_HEIGHT // 2
                
                distance = math.sqrt((x - x2)**2 + (y - y2)**2)
                if distance < 100:
                    alpha = int(255 * (1 - distance / 100) * 0.3)
                    if alpha > 20:
                        connection_color = (*self.colors['accent_secondary'][:3], alpha)
                        self.draw_glowing_line((int(x), int(y)), (int(x2), int(y2)), connection_color, 1)
    
    def render_main_menu(self):
        """Render the main menu screen."""
        # Title with glow effect
        title_text = "🧠 AI ANALYTICS PLATFORM"
        self.render_glowing_text(title_text, self.SCREEN_WIDTH//2, 150, 
                                self.fonts['title'], self.colors['text_primary'], 
                                self.colors['glow'])
        
        # Subtitle
        subtitle_text = "Gamified Data Science Experience"
        self.fonts['heading'].render_to(self.screen, 
                                       (self.SCREEN_WIDTH//2 - 200, 200), 
                                       subtitle_text, self.colors['text_secondary'])
        
        # Version info
        version_text = f"Version 2.0 | Level {self.level} | XP: {self.experience_points}"
        self.fonts['small'].render_to(self.screen, 
                                     (self.SCREEN_WIDTH//2 - 100, 250), 
                                     version_text, self.colors['text_secondary'])
        
        # Render menu buttons
        for button in self.buttons['main_menu']:
            button.render(self.screen, self.fonts['body'])
        
        # Render statistics panel
        self.render_main_menu_stats()
    
    def render_main_menu_stats(self):
        """Render statistics panel on main menu."""
        panel_x, panel_y = 100, 700
        panel_width, panel_height = 500, 200
        
        # Draw panel background
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((*self.colors['bg_secondary'], 180))
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Panel border
        pygame.draw.rect(self.screen, self.colors['accent_primary'], 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Statistics content
        stats = self.get_player_statistics()
        y_offset = panel_y + 20
        
        self.fonts['heading'].render_to(self.screen, (panel_x + 20, y_offset), 
                                       "📊 Your Statistics", self.colors['text_primary'])
        y_offset += 40
        
        for stat_name, stat_value in stats.items():
            self.fonts['body'].render_to(self.screen, (panel_x + 20, y_offset), 
                                        f"{stat_name}: {stat_value}", self.colors['text_secondary'])
            y_offset += 25
    
    def render_data_analysis(self):
        """Render the data analysis screen."""
        # Title bar
        self.render_glowing_text("🔬 DATA ANALYSIS LAB", 200, 50, 
                                self.fonts['heading'], self.colors['text_primary'], 
                                self.colors['accent_primary'])
        
        # Render side panel buttons
        for button in self.buttons['data_analysis']:
            button.render(self.screen, self.fonts['body'])
        
        # Render main panels
        for panel in self.panels.values():
            panel.render(self.screen, self.fonts)
        
        # Render data visualization area
        self.render_data_visualization_area()
        
        # Render real-time progress indicators
        self.render_progress_indicators()
    
    def render_data_visualization_area(self):
        """Render the main data visualization area."""
        viz_x, viz_y = 300, 170
        viz_width, viz_height = 1000, 500
        
        # Background
        viz_surface = pygame.Surface((viz_width, viz_height), pygame.SRCALPHA)
        viz_surface.fill((*self.colors['bg_primary'], 200))
        self.screen.blit(viz_surface, (viz_x, viz_y))
        
        # Border with glow effect
        pygame.draw.rect(self.screen, self.colors['accent_primary'], 
                        (viz_x, viz_y, viz_width, viz_height), 3)
        
        # Render current visualization
        if hasattr(self.main_app, 'current_data') and self.main_app.current_data is not None:
            self.render_interactive_data_viz(viz_x, viz_y, viz_width, viz_height)
        else:
            # No data loaded message
            no_data_text = "🔍 Load data to begin interactive visualization"
            text_rect = self.fonts['heading'].get_rect(no_data_text)
            text_x = viz_x + (viz_width - text_rect.width) // 2
            text_y = viz_y + (viz_height - text_rect.height) // 2
            
            self.fonts['heading'].render_to(self.screen, (text_x, text_y), 
                                           no_data_text, self.colors['text_secondary'])
    
    def render_interactive_data_viz(self, x, y, width, height):
        """Render interactive data visualization."""
        if not NUMPY_AVAILABLE:
            # Fallback visualization
            self.render_simple_data_viz(x, y, width, height)
            return
        
        try:
            data = self.main_app.current_data
            
            # Create node-link visualization for data columns
            center_x, center_y = x + width // 2, y + height // 2
            radius = min(width, height) // 3
            
            num_columns = min(len(data.columns), 20)  # Limit for performance
            angle_step = 2 * math.pi / num_columns
            
            for i, column in enumerate(data.columns[:num_columns]):
                angle = i * angle_step + time.time() * 0.2  # Rotating
                
                node_x = center_x + radius * math.cos(angle)
                node_y = center_y + radius * math.sin(angle)
                
                # Node size based on data completeness
                completeness = 1 - (data[column].isnull().sum() / len(data))
                node_size = 5 + int(completeness * 15)
                
                # Node color based on data type
                if data[column].dtype in ['int64', 'float64']:
                    color = self.colors['accent_primary']
                elif data[column].dtype == 'object':
                    color = self.colors['accent_secondary']
                else:
                    color = self.colors['warning']
                
                # Draw node with glow
                for glow_radius in range(node_size + 5, node_size - 1, -1):
                    alpha = max(0, 50 - (glow_radius - node_size) * 10)
                    glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                    pygame.draw.circle(glow_surface, (*color, alpha), (glow_radius, glow_radius), glow_radius)
                    self.screen.blit(glow_surface, (int(node_x - glow_radius), int(node_y - glow_radius)))
                
                # Column name label
                if node_size > 10:  # Only show label for significant nodes
                    label = column[:10] + "..." if len(column) > 10 else column
                    self.fonts['small'].render_to(self.screen, (int(node_x - 30), int(node_y + 20)), 
                                                 label, self.colors['text_primary'])
            
            # Draw connections between correlated columns
            if hasattr(self.main_app, 'cleaned_data') and self.main_app.cleaned_data is not None:
                numeric_data = self.main_app.cleaned_data.select_dtypes(include=['number'])
                if len(numeric_data.columns) > 1:
                    correlation_matrix = numeric_data.corr().abs()
                    
                    for i, col1 in enumerate(numeric_data.columns[:num_columns]):
                        for j, col2 in enumerate(numeric_data.columns[:num_columns]):
                            if i < j and abs(correlation_matrix.loc[col1, col2]) > 0.5:
                                angle1 = i * angle_step + time.time() * 0.2
                                angle2 = j * angle_step + time.time() * 0.2
                                
                                x1 = center_x + radius * math.cos(angle1)
                                y1 = center_y + radius * math.sin(angle1)
                                x2 = center_x + radius * math.cos(angle2)
                                y2 = center_y + radius * math.sin(angle2)
                                
                                correlation_strength = abs(correlation_matrix.loc[col1, col2])
                                alpha = int(correlation_strength * 150)
                                
                                self.draw_glowing_line((int(x1), int(y1)), (int(x2), int(y2)), 
                                                     (*self.colors['glow'], alpha), 2)
                                
        except Exception as e:
            print(f"Error in interactive data visualization: {e}")
            self.render_simple_data_viz(x, y, width, height)
    
    def render_simple_data_viz(self, x, y, width, height):
        """Render simple fallback data visualization."""
        # Simple bar chart representation
        if hasattr(self.main_app, 'current_data') and self.main_app.current_data is not None:
            data = self.main_app.current_data
            
            # Show basic dataset info
            info_y = y + 50
            self.fonts['body'].render_to(self.screen, (x + 50, info_y), 
                                        f"Dataset Shape: {data.shape[0]:,} rows × {data.shape[1]} columns", 
                                        self.colors['text_primary'])
            
            # Show column types
            info_y += 30
            numeric_cols = len(data.select_dtypes(include=['number']).columns)
            object_cols = len(data.select_dtypes(include=['object']).columns)
            
            self.fonts['body'].render_to(self.screen, (x + 50, info_y), 
                                        f"Numeric Columns: {numeric_cols} | Text Columns: {object_cols}", 
                                        self.colors['text_secondary'])
            
            # Show missing values
            info_y += 30
            missing_percentage = (data.isnull().sum().sum() / (data.shape[0] * data.shape[1])) * 100
            self.fonts['body'].render_to(self.screen, (x + 50, info_y), 
                                        f"Missing Values: {missing_percentage:.1f}%", 
                                        self.colors['warning'] if missing_percentage > 10 else self.colors['success'])
    
    def render_progress_indicators(self):
        """Render real-time progress indicators."""
        # Progress bars for various operations
        progress_y = 750
        
        for operation, progress_info in self.progress_bars.items():
            progress_width = 300
            progress_height = 20
            progress_x = 50
            
            # Background
            pygame.draw.rect(self.screen, self.colors['bg_secondary'], 
                           (progress_x, progress_y, progress_width, progress_height))
            
            # Progress fill
            fill_width = int(progress_width * progress_info['progress'])
            if fill_width > 0:
                color = self.colors['success'] if progress_info['progress'] >= 1.0 else self.colors['accent_primary']
                pygame.draw.rect(self.screen, color, 
                               (progress_x, progress_y, fill_width, progress_height))
            
            # Label
            self.fonts['small'].render_to(self.screen, (progress_x, progress_y - 20), 
                                         f"{operation}: {progress_info['progress']*100:.1f}%", 
                                         self.colors['text_primary'])
            
            progress_y += 40
    
    def render_achievements(self):
        """Render achievements screen."""
        self.render_glowing_text("🏆 ACHIEVEMENTS", self.SCREEN_WIDTH//2, 100, 
                                self.fonts['title'], self.colors['achievement'], 
                                self.colors['glow'])
        
        # Achievement grid
        achievement_data = self.get_achievement_data()
        
        grid_x, grid_y = 200, 200
        grid_cols = 4
        achievement_width, achievement_height = 300, 150
        
        for i, achievement in enumerate(achievement_data):
            col = i % grid_cols
            row = i // grid_cols
            
            ach_x = grid_x + col * (achievement_width + 20)
            ach_y = grid_y + row * (achievement_height + 20)
            
            # Achievement panel
            self.render_achievement_panel(ach_x, ach_y, achievement_width, achievement_height, achievement)
        
        # Back button
        back_button = Button(50, 50, 100, 40, "🔙 Back", self.colors['text_secondary'], 
                           lambda: self.transition_to_screen("main_menu"))
        back_button.render(self.screen, self.fonts['body'])
    
    def render_achievement_panel(self, x, y, width, height, achievement):
        """Render individual achievement panel."""
        # Background
        alpha = 200 if achievement['unlocked'] else 100
        panel_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        panel_surface.fill((*self.colors['bg_secondary'], alpha))
        self.screen.blit(panel_surface, (x, y))
        
        # Border
        border_color = self.colors['achievement'] if achievement['unlocked'] else self.colors['text_secondary']
        pygame.draw.rect(self.screen, border_color, (x, y, width, height), 3)
        
        # Achievement icon
        icon_text = achievement['icon']
        icon_size = 48 if achievement['unlocked'] else 32
        self.render_text(icon_text, x + 20, y + 20, {'size': icon_size}, self.colors['text_primary'])
        
        # Achievement name
        name_color = self.colors['achievement'] if achievement['unlocked'] else self.colors['text_secondary']
        self.fonts['heading'].render_to(self.screen, (x + 80, y + 20), 
                                       achievement['name'], name_color)
        
        # Description
        self.fonts['body'].render_to(self.screen, (x + 20, y + 60), 
                                    achievement['description'], self.colors['text_secondary'])
        
        # Progress bar (if not unlocked)
        if not achievement['unlocked'] and 'progress' in achievement:
            progress_width = width - 40
            progress_height = 10
            progress_x, progress_y = x + 20, y + height - 30
            
            pygame.draw.rect(self.screen, self.colors['bg_primary'], 
                           (progress_x, progress_y, progress_width, progress_height))
            
            fill_width = int(progress_width * achievement['progress'])
            if fill_width > 0:
                pygame.draw.rect(self.screen, self.colors['accent_primary'], 
                               (progress_x, progress_y, fill_width, progress_height))
    
    def render_settings(self):
        """Render settings screen."""
        self.render_glowing_text("⚙️ SETTINGS", self.SCREEN_WIDTH//2, 100, 
                                self.fonts['title'], self.colors['text_primary'], 
                                self.colors['accent_primary'])
        
        # Settings options
        settings_y = 200
        settings_options = [
            "🔊 Sound Effects: ON",
            "🎵 Background Music: OFF",
            "✨ Particle Effects: HIGH",
            "🖥️ Resolution: 1600x1000",
            "🎨 Theme: Cyberpunk Blue",
            "⚡ Performance Mode: OFF"
        ]
        
        for option in settings_options:
            self.fonts['body'].render_to(self.screen, (self.SCREEN_WIDTH//2 - 200, settings_y), 
                                        option, self.colors['text_secondary'])
            settings_y += 40
        
        # Back button
        back_button = Button(50, 50, 100, 40, "🔙 Back", self.colors['text_secondary'], 
                           lambda: self.transition_to_screen("main_menu"))
        back_button.render(self.screen, self.fonts['body'])
    
    def render_particles(self):
        """Render all particle systems."""
        for particle_type, particles in self.particle_systems.items():
            for particle in particles:
                particle.render(self.screen)
    
    def render_ui_overlays(self):
        """Render UI overlays like notifications and tooltips."""
        # Status messages
        message_y = self.SCREEN_HEIGHT - 100
        for message in self.status_messages[-3:]:  # Show last 3 messages
            self.fonts['body'].render_to(self.screen, (50, message_y), 
                                        message['text'], message['color'])
            message_y += 25
        
        # XP and level display
        xp_text = f"Level {self.level} | XP: {self.experience_points}"
        self.fonts['body'].render_to(self.screen, (self.SCREEN_WIDTH - 250, 20), 
                                    xp_text, self.colors['achievement'])
    
    def render_transition(self):
        """Render screen transition effects."""
        if self.transition_alpha > 0:
            transition_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            transition_surface.fill((0, 0, 0, int(self.transition_alpha * 255)))
            self.screen.blit(transition_surface, (0, 0))
    
    # Utility rendering methods
    def render_glowing_text(self, text, x, y, font, color, glow_color):
        """Render text with glow effect."""
        # Render glow layers
        for offset in range(5, 0, -1):
            alpha = 50 - offset * 8
            glow_surface = pygame.Surface(font.get_rect(text).size, pygame.SRCALPHA)
            font.render_to(glow_surface, (0, 0), text, (*glow_color, alpha))
            
            for dx in range(-offset, offset + 1):
                for dy in range(-offset, offset + 1):
                    if dx*dx + dy*dy <= offset*offset:
                        self.screen.blit(glow_surface, (x + dx - font.get_rect(text).width//2, y + dy))
        
        # Render main text
        font.render_to(self.screen, (x - font.get_rect(text).width//2, y), text, color)
    
    def render_text(self, text, x, y, font_config, color):
        """Render text with custom font configuration."""
        # This is a simplified version - you'd implement font scaling here
        self.fonts['body'].render_to(self.screen, (x, y), text, color)
    
    def draw_glowing_line(self, start_pos, end_pos, color, thickness=2):
        """Draw a line with glow effect."""
        # Draw glow layers
        for i in range(thickness + 4, thickness - 1, -1):
            alpha = max(20, 100 - (i - thickness) * 20)
            glow_color = (*color[:3], alpha)
            
            # Create surface for glow
            glow_surf = pygame.Surface((abs(end_pos[0] - start_pos[0]) + i*2, 
                                      abs(end_pos[1] - start_pos[1]) + i*2), pygame.SRCALPHA)
            
            # Calculate relative positions
            rel_start = (i, i)
            rel_end = (end_pos[0] - start_pos[0] + i, end_pos[1] - start_pos[1] + i)
            
            pygame.draw.line(glow_surf, glow_color, rel_start, rel_end, i)
            self.screen.blit(glow_surf, (min(start_pos[0], end_pos[0]) - i, 
                                        min(start_pos[1], end_pos[1]) - i))
        
        # Draw main line
        pygame.draw.line(self.screen, color[:3], start_pos, end_pos, thickness)
    
    # Update methods
    def update_transition(self, dt):
        """Update screen transition animation."""
        transition_speed = 2.0  # seconds
        self.transition_alpha += dt / transition_speed
        
        if self.transition_alpha >= 1.0:
            self.transition_alpha = 0
            self.transitioning = False
    
    def update_particles(self, dt):
        """Update all particle systems."""
        for particle_type, particles in self.particle_systems.items():
            for particle in particles[:]:  # Copy list to allow removal
                particle.update(dt)
                if particle.is_dead():
                    particles.remove(particle)
    
    def update_animations(self, dt):
        """Update UI animations."""
        for animation in self.animations[:]:
            animation.update(dt)
            if animation.is_complete():
                self.animations.remove(animation)
    
    def update_progress_bars(self, dt):
        """Update progress bar animations."""
        for operation, progress_info in self.progress_bars.items():
            # Smooth progress animation
            target_progress = progress_info.get('target_progress', progress_info['progress'])
            current_progress = progress_info['progress']
            
            if abs(target_progress - current_progress) > 0.001:
                progress_info['progress'] += (target_progress - current_progress) * dt * 3
    
    def update_data_visualization(self, dt):
        """Update data visualization animations."""
        # Update any animated elements in the data visualization
        pass
    
    def process_main_app_commands(self):
        """Process commands from the main application."""
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                self.handle_main_app_result(result)
        except queue.Empty:
            pass
    
    def handle_main_app_result(self, result):
        """Handle results from main application operations."""
        if result['type'] == 'data_loaded':
            self.add_status_message("📊 Data loaded successfully!", self.colors['success'])
            self.award_experience(50)
        
        elif result['type'] == 'analysis_complete':
            self.add_status_message("🧠 AI analysis complete!", self.colors['success'])
            self.award_experience(100)
        
        elif result['type'] == 'cleaning_complete':
            self.add_status_message("🧹 Data cleaning finished!", self.colors['success'])
            self.award_experience(75)
        
        elif result['type'] == 'training_complete':
            self.add_status_message("🤖 Model training finished!", self.colors['achievement'])
            self.award_experience(200)
            self.create_success_burst()
        
        elif result['type'] == 'error':
            self.add_status_message(f"❌ Error: {result['message']}", self.colors['error'])
    
    # Event handlers for buttons
    def start_new_analysis(self):
        """Start new analysis workflow."""
        self.transition_to_screen("data_analysis")
        self.add_status_message("🚀 Starting new analysis session...", self.colors['accent_primary'])
    
    def load_session(self):
        """Load existing session."""
        # This would open a file dialog or session browser
        self.add_status_message("📂 Loading session browser...", self.colors['accent_secondary'])
    
    def show_achievements(self):
        """Show achievements screen."""
        self.transition_to_screen("achievements")
    
    def show_settings(self):
        """Show settings screen."""
        self.transition_to_screen("settings")
    
    def exit_to_desktop(self):
        """Exit to desktop (return to main app)."""
        self.running = False
    
    def return_to_menu(self):
        """Return to main menu."""
        self.transition_to_screen("main_menu")
    
    def load_data_dialog(self):
        """Trigger data loading dialog."""
        self.command_queue.put({'action': 'load_data'})
        self.add_status_message("📁 Opening file dialog...", self.colors['accent_primary'])
    
    def run_ai_analysis(self):
        """Run AI analysis on loaded data."""
        if hasattr(self.main_app, 'current_data') and self.main_app.current_data is not None:
            self.command_queue.put({'action': 'analyze_data'})
            self.set_progress('AI Analysis', 0.0)
            self.add_status_message("🧠 Starting AI analysis...", self.colors['accent_primary'])
        else:
            self.add_status_message("⚠️ Please load data first!", self.colors['warning'])
    
    def clean_data(self):
        """Trigger data cleaning."""
        if hasattr(self.main_app, 'current_data') and self.main_app.current_data is not None:
            self.command_queue.put({'action': 'clean_data'})
            self.set_progress('Data Cleaning', 0.0)
            self.add_status_message("🧹 Cleaning data...", self.colors['warning'])
        else:
            self.add_status_message("⚠️ Please load data first!", self.colors['warning'])
    
    def train_models(self):
        """Trigger model training."""
        if hasattr(self.main_app, 'cleaned_data') and self.main_app.cleaned_data is not None:
            self.command_queue.put({'action': 'train_models'})
            self.set_progress('Model Training', 0.0)
            self.add_status_message("🤖 Training models...", self.colors['accent_secondary'])
        else:
            self.add_status_message("⚠️ Please clean data first!", self.colors['warning'])
    
    def create_visualizations(self):
        """Create data visualizations."""
        self.command_queue.put({'action': 'create_visualizations'})
        self.add_status_message("📈 Creating visualizations...", self.colors['achievement'])
    
    def explain_models(self):
        """Generate model explanations."""
        if hasattr(self.main_app, 'training_complete') and self.main_app.training_complete:
            self.command_queue.put({'action': 'explain_models'})
            self.add_status_message("🔍 Generating explanations...", self.colors['glow'])
        else:
            self.add_status_message("⚠️ Please train models first!", self.colors['warning'])
    
    # Utility methods
    def transition_to_screen(self, screen_name):
        """Transition to a different screen."""
        self.current_screen = screen_name
        self.transitioning = True
        self.transition_alpha = 0
    
    def add_status_message(self, message, color):
        """Add a status message."""
        self.status_messages.append({
            'text': message,
            'color': color,
            'timestamp': time.time()
        })
        
        # Keep only last 10 messages
        if len(self.status_messages) > 10:
            self.status_messages.pop(0)
    
    def set_progress(self, operation, progress):
        """Set progress for an operation."""
        self.progress_bars[operation] = {
            'progress': progress,
            'target_progress': progress
        }
    
    def update_progress(self, operation, progress):
        """Update progress for an operation."""
        if operation in self.progress_bars:
            self.progress_bars[operation]['target_progress'] = progress
    
    def award_experience(self, xp):
        """Award experience points and check for level up."""
        self.experience_points += xp
        
        # Check for level up
        xp_needed = self.level * 1000  # 1000 XP per level
        if self.experience_points >= xp_needed:
            self.level += 1
            self.experience_points -= xp_needed
            self.create_level_up_effect()
            self.add_status_message(f"🎉 LEVEL UP! Now level {self.level}!", self.colors['achievement'])
    
    def create_click_effect(self, pos):
        """Create visual effect for button clicks."""
        for i in range(10):
            particle = ClickParticle(pos[0], pos[1], self.colors['accent_primary'])
            self.particle_systems['success_burst'].append(particle)
    
    def create_success_burst(self):
        """Create success particle burst."""
        center_x, center_y = self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2
        for i in range(20):
            particle = SuccessParticle(center_x, center_y, self.colors['achievement'])
            self.particle_systems['success_burst'].append(particle)
    
    def create_level_up_effect(self):
        """Create level up particle effect."""
        for i in range(50):
            particle = LevelUpParticle(self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2, 
                                     self.colors['achievement'])
            self.particle_systems['level_up'].append(particle)
    
    def handle_data_visualization_click(self, pos):
        """Handle clicks in the data visualization area."""
        viz_x, viz_y = 300, 170
        viz_width, viz_height = 1000, 500
        
        if viz_x <= pos[0] <= viz_x + viz_width and viz_y <= pos[1] <= viz_y + viz_height:
            # Clicked in visualization area
            self.add_status_message("🔍 Exploring data point...", self.colors['glow'])
    
    def show_help_overlay(self):
        """Show help overlay."""
        # This would show keyboard shortcuts and help text
        pass
    
    def get_player_statistics(self):
        """Get player statistics for display."""
        return {
            "Sessions Created": getattr(self.main_app.storage_manager, 'storage_stats', {}).get('sessions_created', 0),
            "Models Trained": getattr(self.main_app.storage_manager, 'storage_stats', {}).get('models_saved', 0),
            "Datasets Processed": getattr(self.main_app.storage_manager, 'storage_stats', {}).get('data_processed', 0),
            "Achievements": len([a for a in self.storage_achievements.values() if a.get('unlocked', False)])
        }
    
    def get_achievement_data(self):
        """Get achievement data for display."""
        achievements = [
            {
                'name': 'Data Explorer',
                'description': 'Load your first dataset',
                'icon': '🔍',
                'unlocked': True,  # Example
                'progress': 1.0
            },
            {
                'name': 'AI Apprentice',
                'description': 'Complete your first AI analysis',
                'icon': '🧠',
                'unlocked': True,
                'progress': 1.0
            },
            {
                'name': 'Data Cleaner',
                'description': 'Clean 5 datasets',
                'icon': '🧹',
                'unlocked': False,
                'progress': 0.6
            },
            {
                'name': 'Model Master',
                'description': 'Train 10 machine learning models',
                'icon': '🤖',
                'unlocked': False,
                'progress': 0.3
            },
            {
                'name': 'Visualization Wizard',
                'description': 'Create 25 visualizations',
                'icon': '📊',
                'unlocked': False,
                'progress': 0.8
            },
            {
                'name': 'Insight Hunter',
                'description': 'Generate 50 model explanations',
                'icon': '🔍',
                'unlocked': False,
                'progress': 0.2
            }
        ]
        return achievements
    
    def cleanup(self):
        """Clean up resources before exit."""
        pygame.quit()


# UI Component Classes
class Button:
    """Animated button class."""
    
    def __init__(self, x, y, width, height, text, color, callback):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.callback = callback
        self.hovered = False
        self.hover_scale = 1.0
        self.target_scale = 1.0
    
    def update_hover(self, mouse_pos):
        """Update hover state."""
        self.hovered = self.rect.collidepoint(mouse_pos)
        self.target_scale = 1.1 if self.hovered else 1.0
    
    def is_clicked(self, mouse_pos):
        """Check if button was clicked."""
        return self.rect.collidepoint(mouse_pos)
    
    def execute(self):
        """Execute button callback."""
        if self.callback:
            self.callback()
    
    def render(self, screen, font):
        """Render the button."""
        # Smooth scale animation
        self.hover_scale += (self.target_scale - self.hover_scale) * 0.1
        
        # Calculate scaled rect
        scaled_width = int(self.rect.width * self.hover_scale)
        scaled_height = int(self.rect.height * self.hover_scale)
        scaled_x = self.rect.centerx - scaled_width // 2
        scaled_y = self.rect.centery - scaled_height // 2
        scaled_rect = pygame.Rect(scaled_x, scaled_y, scaled_width, scaled_height)
        
        # Button background with glow effect if hovered
        if self.hovered:
            # Glow effect
            glow_surface = pygame.Surface((scaled_width + 20, scaled_height + 20), pygame.SRCALPHA)
            pygame.draw.rect(glow_surface, (*self.color, 50), 
                           (10, 10, scaled_width, scaled_height), border_radius=10)
            screen.blit(glow_surface, (scaled_x - 10, scaled_y - 10))
        
        # Main button
        pygame.draw.rect(screen, self.color, scaled_rect, border_radius=8)
        pygame.draw.rect(screen, (255, 255, 255), scaled_rect, 3, border_radius=8)
        
        # Button text
        text_rect = font.get_rect(self.text)
        text_x = scaled_rect.centerx - text_rect.width // 2
        text_y = scaled_rect.centery - text_rect.height // 2
        font.render_to(screen, (text_x, text_y), self.text, (255, 255, 255))


class ProgressPanel:
    """Progress panel for showing analysis progress."""
    
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.progress_items = []
    
    def render(self, screen, fonts):
        """Render the progress panel."""
        # Panel background
        panel_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        panel_surface.fill((30, 41, 59, 180))
        screen.blit(panel_surface, self.rect)
        
        # Panel border
        pygame.draw.rect(screen, (59, 130, 246), self.rect, 2)
        
        # Title
        fonts['heading'].render_to(screen, (self.rect.x + 20, self.rect.y + 10), 
                                  "⚡ Analysis Progress", (248, 250, 252))


class DataPreviewPanel:
    """Panel for data preview and information."""
    
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
    
    def render(self, screen, fonts):
        """Render the data preview panel."""
        # Panel background
        panel_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        panel_surface.fill((30, 41, 59, 180))
        screen.blit(panel_surface, self.rect)
        
        # Panel border
        pygame.draw.rect(screen, (147, 51, 234), self.rect, 2)
        
        # Title
        fonts['heading'].render_to(screen, (self.rect.x + 20, self.rect.y + 10), 
                                  "📊 Data Preview", (248, 250, 252))


class ModelPerformancePanel:
    """Panel for model performance metrics."""
    
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
    
    def render(self, screen, fonts):
        """Render the model performance panel."""
        # Panel background
        panel_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        panel_surface.fill((30, 41, 59, 180))
        screen.blit(panel_surface, self.rect)
        
        # Panel border
        pygame.draw.rect(screen, (34, 197, 94), self.rect, 2)
        
        # Title
        fonts['heading'].render_to(screen, (self.rect.x + 20, self.rect.y + 10), 
                                  "🤖 Model Performance", (248, 250, 252))


class AchievementPanel:
    """Panel for showing recent achievements."""
    
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
    
    def render(self, screen, fonts):
        """Render the achievement panel."""
        # Panel background
        panel_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        panel_surface.fill((30, 41, 59, 180))
        screen.blit(panel_surface, self.rect)
        
        # Panel border
        pygame.draw.rect(screen, (255, 215, 0), self.rect, 2)
        
        # Title
        fonts['heading'].render_to(screen, (self.rect.x + 20, self.rect.y + 10), 
                                  "🏆 Recent Achievements", (248, 250, 252))


# Particle Classes
class Particle:
    """Base particle class."""
    
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.life = 1.0
        self.max_life = 1.0
    
    def update(self, dt):
        """Update particle."""
        self.life -= dt
    
    def is_dead(self):
        """Check if particle is dead."""
        return self.life <= 0
    
    def render(self, screen):
        """Render particle."""
        alpha = int(255 * (self.life / self.max_life))
        if alpha > 0:
            color = (*self.color[:3], alpha)
            pygame.draw.circle(screen, color[:3], (int(self.x), int(self.y)), 3)


class ClickParticle(Particle):
    """Particle for click effects."""
    
    def __init__(self, x, y, color):
        super().__init__(x, y, color)
        self.vel_x = (random.random() - 0.5) * 100
        self.vel_y = (random.random() - 0.5) * 100
        self.max_life = 0.5
        self.life = self.max_life
    
    def update(self, dt):
        super().update(dt)
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt
        self.vel_x *= 0.95  # Friction
        self.vel_y *= 0.95


class SuccessParticle(Particle):
    """Particle for success effects."""
    
    def __init__(self, x, y, color):
        super().__init__(x, y, color)
        angle = random.random() * 2 * math.pi
        speed = random.random() * 200 + 50
        self.vel_x = math.cos(angle) * speed
        self.vel_y = math.sin(angle) * speed
        self.max_life = 2.0
        self.life = self.max_life
    
    def update(self, dt):
        super().update(dt)
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt
        self.vel_y += 100 * dt  # Gravity


class LevelUpParticle(Particle):
    """Particle for level up effects."""
    
    def __init__(self, x, y, color):
        super().__init__(x, y, color)
        angle = random.random() * 2 * math.pi
        speed = random.random() * 300 + 100
        self.vel_x = math.cos(angle) * speed
        self.vel_y = math.sin(angle) * speed - 150  # Upward bias
        self.max_life = 3.0
        self.life = self.max_life
        self.size = random.randint(5, 15)
    
    def update(self, dt):
        super().update(dt)
        self.x += self.vel_x * dt
        self.y += self.vel_y * dt
        self.vel_y += 50 * dt  # Light gravity
    
    def render(self, screen):
        alpha = int(255 * (self.life / self.max_life))
        if alpha > 0:
            color = (*self.color[:3], alpha)
            pygame.draw.circle(screen, color[:3], (int(self.x), int(self.y)), self.size)


# Animation Classes
class Animation:
    """Base animation class."""
    
    def __init__(self, duration):
        self.duration = duration
        self.current_time = 0.0
        self.complete = False
    
    def update(self, dt):
        """Update animation."""
        self.current_time += dt
        if self.current_time >= self.duration:
            self.complete = True
            self.current_time = self.duration
    
    def is_complete(self):
        """Check if animation is complete."""
        return self.complete
    
    def get_progress(self):
        """Get animation progress (0.0 to 1.0)."""
        return min(1.0, self.current_time / self.duration)


class FadeAnimation(Animation):
    """Fade in/out animation."""
    
    def __init__(self, duration, fade_in=True):
        super().__init__(duration)
        self.fade_in = fade_in
    
    def get_alpha(self):
        """Get current alpha value."""
        progress = self.get_progress()
        if self.fade_in:
            return int(255 * progress)
        else:
            return int(255 * (1.0 - progress))


class SlideAnimation(Animation):
    """Slide animation for UI elements."""
    
    def __init__(self, duration, start_pos, end_pos):
        super().__init__(duration)
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def get_position(self):
        """Get current position."""
        progress = self.get_progress()
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress
        return (int(x), int(y))


class PulseAnimation(Animation):
    """Pulse animation for highlighting elements."""
    
    def __init__(self, duration, min_scale=0.8, max_scale=1.2):
        super().__init__(duration)
        self.min_scale = min_scale
        self.max_scale = max_scale
    
    def get_scale(self):
        """Get current scale value."""
        progress = self.get_progress()
        # Use sine wave for smooth pulsing
        pulse_factor = math.sin(progress * math.pi * 2)
        scale_range = self.max_scale - self.min_scale
        return self.min_scale + (scale_range * (pulse_factor + 1) / 2)


# Enhanced UI Components
class InteractivePanel:
    """Base class for interactive panels."""
    
    def __init__(self, x, y, width, height, title="Panel"):
        self.rect = pygame.Rect(x, y, width, height)
        self.title = title
        self.visible = True
        self.dragging = False
        self.drag_offset = (0, 0)
        self.animations = []
        self.child_elements = []
    
    def handle_event(self, event):
        """Handle pygame events for this panel."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                if event.button == 1:  # Left click
                    self.dragging = True
                    self.drag_offset = (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y)
                return True
        
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.dragging = False
        
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                self.rect.x = event.pos[0] - self.drag_offset[0]
                self.rect.y = event.pos[1] - self.drag_offset[1]
                return True
        
        return False
    
    def update(self, dt):
        """Update panel animations."""
        for animation in self.animations[:]:
            animation.update(dt)
            if animation.is_complete():
                self.animations.remove(animation)
    
    def render(self, screen, fonts, colors):
        """Render the panel."""
        if not self.visible:
            return
        
        # Panel background with transparency
        panel_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        panel_surface.fill((*colors['bg_secondary'], 200))
        screen.blit(panel_surface, self.rect)
        
        # Panel border with glow effect
        pygame.draw.rect(screen, colors['accent_primary'], self.rect, 3)
        
        # Title bar
        title_rect = pygame.Rect(self.rect.x, self.rect.y, self.rect.width, 30)
        pygame.draw.rect(screen, colors['accent_primary'], title_rect)
        
        # Title text
        fonts['body'].render_to(screen, (self.rect.x + 10, self.rect.y + 5), 
                               self.title, colors['text_primary'])
        
        # Render child elements
        for element in self.child_elements:
            element.render(screen, fonts, colors)


class DataVisualizationPanel(InteractivePanel):
    """Interactive data visualization panel."""
    
    def __init__(self, x, y, width, height, data_manager):
        super().__init__(x, y, width, height, "🔬 Data Visualization")
        self.data_manager = data_manager
        self.visualization_type = "scatter"
        self.selected_columns = []
        self.zoom_level = 1.0
        self.pan_offset = (0, 0)
        self.hover_point = None
    
    def set_data(self, data):
        """Set data for visualization."""
        self.data = data
        if data is not None:
            self.selected_columns = list(data.select_dtypes(include=[np.number]).columns[:2])
    
    def handle_event(self, event):
        """Handle visualization-specific events."""
        if super().handle_event(event):
            return True
        
        if not self.rect.collidepoint(pygame.mouse.get_pos()):
            return False
        
        if event.type == pygame.MOUSEWHEEL:
            # Zoom functionality
            zoom_factor = 1.1 if event.y > 0 else 0.9
            self.zoom_level *= zoom_factor
            self.zoom_level = max(0.1, min(5.0, self.zoom_level))
            return True
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 3:  # Right click for context menu
                self.show_context_menu(event.pos)
                return True
        
        return False
    
    def render_visualization(self, screen, fonts, colors):
        """Render the actual data visualization."""
        if not hasattr(self, 'data') or self.data is None:
            # No data message
            no_data_text = "📊 No data loaded"
            text_rect = fonts['body'].get_rect(no_data_text)
            text_x = self.rect.centerx - text_rect.width // 2
            text_y = self.rect.centery - text_rect.height // 2
            fonts['body'].render_to(screen, (text_x, text_y), no_data_text, colors['text_secondary'])
            return
        
        # Visualization area (below title bar)
        viz_rect = pygame.Rect(self.rect.x + 10, self.rect.y + 40, 
                              self.rect.width - 20, self.rect.height - 50)
        
        # Background
        pygame.draw.rect(screen, colors['bg_primary'], viz_rect)
        pygame.draw.rect(screen, colors['accent_secondary'], viz_rect, 1)
        
        if len(self.selected_columns) >= 2:
            self.render_scatter_plot(screen, viz_rect, colors)
        else:
            # Show column selection interface
            self.render_column_selector(screen, fonts, viz_rect, colors)
    
    def render_scatter_plot(self, screen, viz_rect, colors):
        """Render scatter plot visualization."""
        try:
            col_x, col_y = self.selected_columns[0], self.selected_columns[1]
            x_data = self.data[col_x].dropna()
            y_data = self.data[col_y].dropna()
            
            if len(x_data) == 0 or len(y_data) == 0:
                return
            
            # Normalize data to visualization area
            x_min, x_max = x_data.min(), x_data.max()
            y_min, y_max = y_data.min(), y_data.max()
            
            if x_max == x_min:
                x_max = x_min + 1
            if y_max == y_min:
                y_max = y_min + 1
            
            # Sample data for performance (max 1000 points)
            sample_size = min(1000, len(x_data))
            if len(x_data) > sample_size:
                sample_indices = np.random.choice(len(x_data), sample_size, replace=False)
                x_sample = x_data.iloc[sample_indices]
                y_sample = y_data.iloc[sample_indices]
            else:
                x_sample = x_data
                y_sample = y_data
            
            # Plot points
            for i, (x_val, y_val) in enumerate(zip(x_sample, y_sample)):
                # Convert to screen coordinates
                screen_x = viz_rect.x + int(((x_val - x_min) / (x_max - x_min)) * viz_rect.width)
                screen_y = viz_rect.y + int((1 - (y_val - y_min) / (y_max - y_min)) * viz_rect.height)
                
                # Apply zoom and pan
                screen_x = int((screen_x - viz_rect.centerx) * self.zoom_level + viz_rect.centerx + self.pan_offset[0])
                screen_y = int((screen_y - viz_rect.centery) * self.zoom_level + viz_rect.centery + self.pan_offset[1])
                
                # Only draw if point is visible
                if viz_rect.collidepoint(screen_x, screen_y):
                    # Color points based on density or value
                    point_color = colors['accent_primary']
                    if i % 10 == 0:  # Highlight every 10th point
                        point_color = colors['achievement']
                    
                    pygame.draw.circle(screen, point_color, (screen_x, screen_y), 3)
                    
                    # Add glow effect for highlighted points
                    if point_color == colors['achievement']:
                        glow_surface = pygame.Surface((10, 10), pygame.SRCALPHA)
                        pygame.draw.circle(glow_surface, (*colors['glow'], 100), (5, 5), 5)
                        screen.blit(glow_surface, (screen_x - 5, screen_y - 5))
            
            # Draw axes labels
            font_small = pygame.font.Font(None, 20)
            x_label = font_small.render(col_x, True, colors['text_secondary'])
            y_label = font_small.render(col_y, True, colors['text_secondary'])
            
            screen.blit(x_label, (viz_rect.centerx - x_label.get_width() // 2, viz_rect.bottom - 20))
            
            # Rotate y label
            y_label_rotated = pygame.transform.rotate(y_label, 90)
            screen.blit(y_label_rotated, (viz_rect.x + 5, viz_rect.centery - y_label_rotated.get_height() // 2))
            
        except Exception as e:
            print(f"Error rendering scatter plot: {e}")
    
    def render_column_selector(self, screen, fonts, viz_rect, colors):
        """Render column selection interface."""
        fonts['body'].render_to(screen, (viz_rect.x + 10, viz_rect.y + 10), 
                               "Select columns for visualization:", colors['text_primary'])
        
        if hasattr(self, 'data'):
            numeric_cols = list(self.data.select_dtypes(include=[np.number]).columns)
            for i, col in enumerate(numeric_cols[:10]):  # Show first 10 columns
                y_pos = viz_rect.y + 40 + i * 25
                col_text = f"• {col}"
                fonts['small'].render_to(screen, (viz_rect.x + 20, y_pos), 
                                        col_text, colors['text_secondary'])
    
    def show_context_menu(self, pos):
        """Show context menu for visualization options."""
        # This would show a context menu with options like:
        # - Change visualization type
        # - Select different columns
        # - Export visualization
        pass


class ModelPerformanceWidget:
    """Widget for displaying model performance metrics."""
    
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.models_data = {}
        self.selected_metric = "accuracy"
        self.animation_time = 0.0
    
    def set_models_data(self, models_data):
        """Set model performance data."""
        self.models_data = models_data
    
    def update(self, dt):
        """Update animations."""
        self.animation_time += dt
    
    def render(self, screen, fonts, colors):
        """Render model performance widget."""
        # Background
        widget_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        widget_surface.fill((*colors['bg_secondary'], 180))
        screen.blit(widget_surface, self.rect)
        
        # Border
        pygame.draw.rect(screen, colors['success'], self.rect, 2)
        
        # Title
        fonts['heading'].render_to(screen, (self.rect.x + 10, self.rect.y + 10), 
                                  "🤖 Model Performance", colors['text_primary'])
        
        if not self.models_data:
            fonts['body'].render_to(screen, (self.rect.x + 10, self.rect.y + 50), 
                                   "No models trained yet", colors['text_secondary'])
            return
        
        # Performance bars
        y_offset = self.rect.y + 50
        bar_height = 25
        max_width = self.rect.width - 60
        
        for i, (model_name, performance) in enumerate(self.models_data.items()):
            if i >= 5:  # Show max 5 models
                break
            
            # Model name
            fonts['body'].render_to(screen, (self.rect.x + 10, y_offset), 
                                   model_name, colors['text_primary'])
            
            # Performance bar
            bar_rect = pygame.Rect(self.rect.x + 150, y_offset, max_width - 140, bar_height - 5)
            
            # Background bar
            pygame.draw.rect(screen, colors['bg_primary'], bar_rect)
            
            # Performance fill with animation
            performance_value = performance.get(self.selected_metric, 0)
            animated_performance = performance_value * min(1.0, (self.animation_time - i * 0.2))
            
            if animated_performance > 0:
                fill_width = int(bar_rect.width * animated_performance)
                fill_rect = pygame.Rect(bar_rect.x, bar_rect.y, fill_width, bar_rect.height)
                
                # Color based on performance
                if performance_value > 0.9:
                    bar_color = colors['achievement']
                elif performance_value > 0.8:
                    bar_color = colors['success']
                elif performance_value > 0.7:
                    bar_color = colors['warning']
                else:
                    bar_color = colors['error']
                
                pygame.draw.rect(screen, bar_color, fill_rect)
                
                # Performance text
                perf_text = f"{performance_value:.3f}"
                fonts['small'].render_to(screen, (bar_rect.right + 10, y_offset + 5), 
                                        perf_text, colors['text_primary'])
            
            # Border
            pygame.draw.rect(screen, colors['accent_primary'], bar_rect, 1)
            
            y_offset += 35


class QuestSystem:
    """Game-like quest system for data analysis tasks."""
    
    def __init__(self):
        self.active_quests = []
        self.completed_quests = []
        self.available_quests = self._initialize_quests()
    
    def _initialize_quests(self):
        """Initialize available quests."""
        return [
            {
                'id': 'first_data_load',
                'title': '📊 Data Explorer',
                'description': 'Load your first dataset',
                'objectives': ['Load any CSV, Excel, or JSON file'],
                'reward_xp': 100,
                'reward_items': ['Data Explorer Badge'],
                'type': 'tutorial',
                'prerequisites': []
            },
            {
                'id': 'data_analysis_master',
                'title': '🧠 AI Analysis Guru',
                'description': 'Complete comprehensive data analysis',
                'objectives': [
                    'Run AI analysis on dataset',
                    'Identify data quality issues',
                    'Generate cleaning recommendations'
                ],
                'reward_xp': 250,
                'reward_items': ['Analysis Master Certificate'],
                'type': 'analysis',
                'prerequisites': ['first_data_load']
            },
            {
                'id': 'model_trainer',
                'title': '🤖 Machine Learning Apprentice',
                'description': 'Train your first ML model',
                'objectives': [
                    'Clean dataset',
                    'Select target variable',
                    'Train at least 3 different models',
                    'Achieve >80% accuracy'
                ],
                'reward_xp': 400,
                'reward_items': ['ML Trainer Badge', 'Model Architect Title'],
                'type': 'modeling',
                'prerequisites': ['data_analysis_master']
            },
            {
                'id': 'explanation_seeker',
                'title': '🔍 Model Explainer',
                'description': 'Understand your models deeply',
                'objectives': [
                    'Generate SHAP explanations',
                    'Create LIME analysis',
                    'Identify top 5 important features',
                    'Analyze model errors'
                ],
                'reward_xp': 300,
                'reward_items': ['Explainability Expert Badge'],
                'type': 'explainability',
                'prerequisites': ['model_trainer']
            },
            {
                'id': 'visualization_artist',
                'title': '📈 Data Visualization Master',
                'description': 'Create stunning visualizations',
                'objectives': [
                    'Create 5 different chart types',
                    'Use 3D visualization features',
                    'Export high-quality plots',
                    'Create interactive dashboard'
                ],
                'reward_xp': 350,
                'reward_items': ['Visualization Artist Badge', 'Chart Master Title'],
                'type': 'visualization',
                'prerequisites': ['data_analysis_master']
            },
            {
                'id': 'data_scientist_legend',
                'title': '🏆 Data Science Legend',
                'description': 'Master all aspects of data science',
                'objectives': [
                    'Complete 10 different analysis sessions',
                    'Train 25 models with >90% accuracy',
                    'Generate 50 explanations',
                    'Create 100 visualizations',
                    'Achieve Level 10'
                ],
                'reward_xp': 1000,
                'reward_items': ['Legend Status', 'Golden Neural Network Badge', 'Master Data Scientist Title'],
                'type': 'mastery',
                'prerequisites': ['model_trainer', 'explanation_seeker', 'visualization_artist']
            }
        ]
    
    def get_available_quests(self, completed_quest_ids):
        """Get quests that are available to start."""
        available = []
        for quest in self.available_quests:
            if quest['id'] not in completed_quest_ids:
                # Check prerequisites
                prerequisites_met = all(prereq in completed_quest_ids for prereq in quest['prerequisites'])
                if prerequisites_met:
                    available.append(quest)
        return available
    
    def start_quest(self, quest_id):
        """Start a new quest."""
        quest = next((q for q in self.available_quests if q['id'] == quest_id), None)
        if quest:
            quest_instance = {
                **quest,
                'started_at': datetime.now(),
                'progress': 0.0,
                'objectives_completed': [False] * len(quest['objectives'])
            }
            self.active_quests.append(quest_instance)
            return quest_instance
        return None
    
    def update_quest_progress(self, action_type, action_data=None):
        """Update quest progress based on user actions."""
        for quest in self.active_quests:
            self._check_quest_objectives(quest, action_type, action_data)
    
    def _check_quest_objectives(self, quest, action_type, action_data):
        """Check if action completes any quest objectives."""
        quest_type = quest['type']
        objectives = quest['objectives']
        
        if quest_type == 'tutorial' and action_type == 'data_loaded':
            quest['objectives_completed'][0] = True
        
        elif quest_type == 'analysis':
            if action_type == 'ai_analysis_complete':
                quest['objectives_completed'][0] = True
            elif action_type == 'data_quality_issues_identified':
                quest['objectives_completed'][1] = True
            elif action_type == 'cleaning_recommendations_generated':
                quest['objectives_completed'][2] = True
        
        elif quest_type == 'modeling':
            if action_type == 'data_cleaned':
                quest['objectives_completed'][0] = True
            elif action_type == 'target_selected':
                quest['objectives_completed'][1] = True
            elif action_type == 'models_trained' and action_data:
                if action_data.get('model_count', 0) >= 3:
                    quest['objectives_completed'][2] = True
                if action_data.get('best_accuracy', 0) > 0.8:
                    quest['objectives_completed'][3] = True
        
        elif quest_type == 'explainability':
            if action_type == 'shap_generated':
                quest['objectives_completed'][0] = True
            elif action_type == 'lime_generated':
                quest['objectives_completed'][1] = True
            elif action_type == 'top_features_identified':
                quest['objectives_completed'][2] = True
            elif action_type == 'error_analysis_complete':
                quest['objectives_completed'][3] = True
        
        elif quest_type == 'visualization':
            if action_type == 'visualization_created':
                chart_types_created = action_data.get('chart_types_created', 0)
                if chart_types_created >= 5:
                    quest['objectives_completed'][0] = True
            elif action_type == '3d_visualization_used':
                quest['objectives_completed'][1] = True
            elif action_type == 'plot_exported':
                quest['objectives_completed'][2] = True
            elif action_type == 'interactive_dashboard_created':
                quest['objectives_completed'][3] = True
        
        # Update overall progress
        completed_count = sum(quest['objectives_completed'])
        quest['progress'] = completed_count / len(quest['objectives'])
        
        # Check if quest is complete
        if quest['progress'] >= 1.0:
            self._complete_quest(quest)
    
    def _complete_quest(self, quest):
        """Complete a quest and award rewards."""
        quest['completed_at'] = datetime.now()
        self.completed_quests.append(quest)
        if quest in self.active_quests:
            self.active_quests.remove(quest)
        
        return {
            'quest_completed': quest,
            'xp_awarded': quest['reward_xp'],
            'items_awarded': quest['reward_items']
        }


class NotificationSystem:
    """In-game notification system."""
    
    def __init__(self):
        self.notifications = []
        self.max_notifications = 5
    
    def add_notification(self, title, message, notification_type="info", duration=5.0):
        """Add a new notification."""
        notification = {
            'title': title,
            'message': message,
            'type': notification_type,
            'created_at': time.time(),
            'duration': duration,
            'alpha': 0.0,
            'target_alpha': 255.0,
            'y_offset': 0.0,
            'target_y': 0.0
        }
        
        self.notifications.insert(0, notification)
        
        # Update positions
        for i, notif in enumerate(self.notifications):
            notif['target_y'] = i * 80
        
        # Remove old notifications
        if len(self.notifications) > self.max_notifications:
            self.notifications = self.notifications[:self.max_notifications]
    
    def update(self, dt):
        """Update notifications."""
        current_time = time.time()
        
        for notification in self.notifications[:]:
            # Update alpha for fade in/out
            notification['alpha'] += (notification['target_alpha'] - notification['alpha']) * dt * 5
            
            # Update position
            notification['y_offset'] += (notification['target_y'] - notification['y_offset']) * dt * 8
            
            # Check if notification should start fading out
            time_alive = current_time - notification['created_at']
            if time_alive > notification['duration'] - 1.0:  # Start fading 1 second before removal
                notification['target_alpha'] = 0.0
            
            # Remove if expired and faded out
            if time_alive > notification['duration'] and notification['alpha'] < 10:
                self.notifications.remove(notification)
                # Update positions of remaining notifications
                for i, notif in enumerate(self.notifications):
                    notif['target_y'] = i * 80
    
    def render(self, screen, fonts, colors):
        """Render notifications."""
        notification_x = screen.get_width() - 320
        base_y = 20
        
        for notification in self.notifications:
            if notification['alpha'] < 5:
                continue
            
            y_pos = base_y + notification['y_offset']
            
            # Notification background
            notif_width, notif_height = 300, 70
            notif_surface = pygame.Surface((notif_width, notif_height), pygame.SRCALPHA)
            
            # Background color based on type
            bg_color = colors['bg_secondary']
            if notification['type'] == 'success':
                accent_color = colors['success']
            elif notification['type'] == 'warning':
                accent_color = colors['warning']
            elif notification['type'] == 'error':
                accent_color = colors['error']
            elif notification['type'] == 'achievement':
                accent_color = colors['achievement']
            else:
                accent_color = colors['accent_primary']
            
            # Draw background with current alpha
            alpha = int(notification['alpha'])
            notif_surface.fill((*bg_color, min(200, alpha)))
            
            # Draw accent border
            pygame.draw.rect(notif_surface, (*accent_color, alpha), (0, 0, notif_width, notif_height), 3)
            
            screen.blit(notif_surface, (notification_x, y_pos))
            
            # Text with alpha
            title_color = (*colors['text_primary'][:3], alpha)
            message_color = (*colors['text_secondary'][:3], alpha)
            
            # Create surfaces for text with alpha
            title_surface = pygame.Surface(fonts['body'].get_rect(notification['title']).size, pygame.SRCALPHA)
            fonts['body'].render_to(title_surface, (0, 0), notification['title'], title_color)
            screen.blit(title_surface, (notification_x + 10, y_pos + 10))
            
            message_surface = pygame.Surface(fonts['small'].get_rect(notification['message']).size, pygame.SRCALPHA)
            fonts['small'].render_to(message_surface, (0, 0), notification['message'], message_color)
            screen.blit(message_surface, (notification_x + 10, y_pos + 35))


# Enhanced Main PygameInterface Updates
def create_enhanced_pygame_interface():
    """Factory function to create enhanced pygame interface."""
    
    class EnhancedPygameInterface(PygameInterface):
        """Enhanced Pygame interface with all gaming features."""
        
        def __init__(self, main_app):
            super().__init__(main_app)
            
            # Enhanced gaming systems
            self.quest_system = QuestSystem()
            self.notification_system = NotificationSystem()
            self.data_viz_panel = None
            self.model_perf_widget = None
            
            # Enhanced state tracking
            self.player_stats = {
                'sessions_completed': 0,
                'models_trained': 0,
                'accuracy_records': [],
                'visualizations_created': 0,
                'explanations_generated': 0,
                'time_played': 0.0,
                'highest_level': 1,
                'total_xp_earned': 0
            }
            
            # Performance tracking
            self.frame_times = []
            self.fps_display = True
        
        def initialize_ui_elements(self):
            """Initialize enhanced UI elements."""
            super().initialize_ui_elements()
            
            # Enhanced data analysis interface
            self.data_viz_panel = DataVisualizationPanel(300, 200, 800, 400, self.main_app)
            self.model_perf_widget = ModelPerformanceWidget(50, 600, 250, 200)
            
            # Add quest panel
            self.panels['quests'] = InteractivePanel(1100, 200, 300, 400, "🎯 Active Quests")
        
        def handle_main_app_result(self, result):
            """Enhanced result handling with quest updates."""
            super().handle_main_app_result(result)
            
            # Update quests based on actions
            if result['type'] == 'data_loaded':
                self.quest_system.update_quest_progress('data_loaded')
                self.notification_system.add_notification(
                    "🎯 Quest Progress!", 
                    "Data loading objective completed!", 
                    "achievement"
                )
            
            elif result['type'] == 'analysis_complete':
                self.quest_system.update_quest_progress('ai_analysis_complete')
                self.quest_system.update_quest_progress('data_quality_issues_identified')
                self.quest_system.update_quest_progress('cleaning_recommendations_generated')
            
            elif result['type'] == 'training_complete':
                model_data = result.get('data', {})
                self.quest_system.update_quest_progress('models_trained', model_data)
                
                # Update model performance widget
                if hasattr(self, 'model_perf_widget'):
                    self.model_perf_widget.set_models_data(model_data.get('models', {}))
            
            elif result['type'] == 'shap_generated':
                self.quest_system.update_quest_progress('shap_generated')
            
            elif result['type'] == 'lime_generated':
                self.quest_system.update_quest_progress('lime_generated')
        
        def update(self, dt):
            """Enhanced update with all gaming systems."""
            super().update(dt)
            
            # Update gaming systems
            self.notification_system.update(dt)
            self.player_stats['time_played'] += dt
            
            # Update enhanced panels
            if self.data_viz_panel:
                self.data_viz_panel.update(dt)
            if self.model_perf_widget:
                self.model_perf_widget.update(dt)
            
            # Performance tracking
            self.frame_times.append(dt)
            if len(self.frame_times) > 60:  # Keep last 60 frames
                self.frame_times.pop(0)
        
        def render_enhanced_ui(self):
            """Render enhanced UI elements."""
            # Render data visualization panel
            if self.data_viz_panel and self.current_screen == "data_analysis":
                self.data_viz_panel.render_visualization(self.screen, self.fonts, self.colors)
            
            # Render model performance widget
            if self.model_perf_widget and self.current_screen == "data_analysis":
                self.model_perf_widget.render(self.screen, self.fonts, self.colors)
            
            # Render notifications
            self.notification_system.render(self.screen, self.fonts, self.colors)
            
            # Render FPS counter if enabled
            if self.fps_display:
                self.render_fps_counter()
        
        def render_fps_counter(self):
            """Render FPS counter."""
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                fps_text = f"FPS: {fps:.1f}"
                self.fonts['small'].render_to(self.screen, (10, 10), fps_text, self.colors['text_secondary'])
        
        def render(self):
            """Enhanced render method."""
            super().render()
            self.render_enhanced_ui()
        
        def show_quest_completion(self, quest_data):
            """Show quest completion celebration."""
            self.notification_system.add_notification(
                "🏆 QUEST COMPLETED!",
                f"{quest_data['quest_completed']['title']} - {quest_data['xp_awarded']} XP!",
                "achievement",
                duration=8.0
            )
            
            # Create celebration particles
            self.create_success_burst()
            self.award_experience(quest_data['xp_awarded'])
        
        def toggle_fps_display(self):
            """Toggle FPS display."""
            self.fps_display = not self.fps_display
        
        def save_player_stats(self):
            """Save player statistics."""
            stats_file = Path("player_stats.json")
            try:
                with open(stats_file, 'w') as f:
                    json.dump(self.player_stats, f, indent=2)
            except Exception as e:
                print(f"Error saving player stats: {e}")
        
        def load_player_stats(self):
            """Load player statistics."""
            stats_file = Path("player_stats.json")
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        self.player_stats.update(json.load(f))
                except Exception as e:
                    print(f"Error loading player stats: {e}")
    
    return EnhancedPygameInterface



