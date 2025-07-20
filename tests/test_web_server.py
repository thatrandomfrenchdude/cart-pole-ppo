"""
Tests for the web server module.
"""
import json
import pytest
import tempfile
import os
from collections import deque
from unittest.mock import patch, mock_open
from web_server import create_app


class TestWebServer:
    
    def test_create_app(self, mock_shared_state):
        """Test Flask app creation."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        # Check that app is created
        assert app is not None
        assert hasattr(app, 'test_client')
    
    def test_state_endpoint(self, mock_shared_state):
        """Test /state endpoint."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        # Set some test values
        current_state.update({
            "position": 0.5,
            "velocity": 1.0,
            "angle": 0.1,
            "angular_velocity": 0.2,
            "reward": 1.0,
            "episode": 5,
            "timestep": 100
        })
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            response = client.get('/state')
            
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['position'] == 0.5
            assert data['velocity'] == 1.0
            assert data['angle'] == 0.1
            assert data['angular_velocity'] == 0.2
            assert data['reward'] == 1.0
            assert data['episode'] == 5
            assert data['timestep'] == 100
    
    def test_history_endpoint_empty(self, mock_shared_state):
        """Test /history endpoint with empty data."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            response = client.get('/history')
            
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['rewards'] == []
            assert data['avg_reward'] == 0.0
    
    def test_history_endpoint_with_data(self, mock_shared_state):
        """Test /history endpoint with data."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        # Add some test data
        test_rewards = [100.0, 150.0, 200.0, 180.0]
        reward_history.extend(test_rewards)
        
        test_episode_rewards = [190.0, 195.0, 200.0]
        episode_rewards.extend(test_episode_rewards)
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            response = client.get('/history')
            
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert data['rewards'] == test_rewards
            # Average should be (190 + 195 + 200) / 3 = 195.0
            assert abs(data['avg_reward'] - 195.0) < 1e-10
    
    @patch("builtins.open", new_callable=mock_open, read_data="<html><body>Test HTML</body></html>")
    def test_index_endpoint(self, mock_file, mock_shared_state):
        """Test / (index) endpoint."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            response = client.get('/')
            
            assert response.status_code == 200
            assert b"Test HTML" in response.data
            
            # Verify file was opened correctly
            mock_file.assert_called_once_with('src/visualization/index.html', 'r')
    
    @patch("builtins.open", new_callable=mock_open, read_data="body { color: blue; }")
    def test_styles_endpoint(self, mock_file, mock_shared_state):
        """Test /styles.css endpoint."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            response = client.get('/styles.css')
            
            assert response.status_code == 200
            assert b"body { color: blue; }" in response.data
            # Check that the content type starts with text/css (may include charset)
            assert response.content_type.startswith('text/css')
    
    @patch("builtins.open", new_callable=mock_open, read_data="console.log('test javascript');")
    def test_visualization_js_endpoint(self, mock_file, mock_shared_state):
        """Test /visualization.js endpoint."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            response = client.get('/visualization.js')
            
            assert response.status_code == 200
            assert b"console.log('test javascript');" in response.data
            # Check that the content type starts with application/javascript (may include charset)
            assert response.content_type.startswith('application/javascript')
    
    def test_state_updates_in_real_time(self, mock_shared_state):
        """Test that state endpoint reflects real-time updates."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            # Initial state
            response = client.get('/state')
            data = json.loads(response.data)
            assert data['position'] == 0
            
            # Update shared state
            current_state['position'] = 1.5
            current_state['episode'] = 10
            
            # Check updated state
            response = client.get('/state')
            data = json.loads(response.data)
            assert data['position'] == 1.5
            assert data['episode'] == 10
    
    def test_history_updates_in_real_time(self, mock_shared_state):
        """Test that history endpoint reflects real-time updates."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            # Initial empty history
            response = client.get('/history')
            data = json.loads(response.data)
            assert len(data['rewards']) == 0
            
            # Add data to shared history
            reward_history.extend([100.0, 200.0])
            episode_rewards.extend([150.0, 175.0])
            
            # Check updated history
            response = client.get('/history')
            data = json.loads(response.data)
            assert len(data['rewards']) == 2
            assert data['rewards'] == [100.0, 200.0]
            assert data['avg_reward'] == 162.5  # (150 + 175) / 2
    
    def test_json_content_type(self, mock_shared_state):
        """Test that JSON endpoints return correct content type."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            # Test state endpoint
            response = client.get('/state')
            assert 'application/json' in response.content_type
            
            # Test history endpoint
            response = client.get('/history')
            assert 'application/json' in response.content_type
    
    def test_large_history_data(self, mock_shared_state):
        """Test handling large amounts of history data."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        # Add large amount of data (within deque limits)
        large_rewards = list(range(100))  # 0 to 99
        large_episode_rewards = list(range(50, 100))  # 50 to 99
        
        reward_history.extend(large_rewards)
        episode_rewards.extend(large_episode_rewards)
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            response = client.get('/history')
            
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert len(data['rewards']) == 100
            assert len(data['rewards']) == len(large_rewards)
            assert data['rewards'] == large_rewards
    
    def test_nonexistent_endpoints(self, mock_shared_state):
        """Test that non-existent endpoints return 404."""
        current_state, reward_history, episode_rewards, _ = mock_shared_state
        
        app = create_app(current_state, reward_history, episode_rewards)
        
        with app.test_client() as client:
            response = client.get('/nonexistent')
            assert response.status_code == 404
            
            response = client.get('/api/unknown')
            assert response.status_code == 404
