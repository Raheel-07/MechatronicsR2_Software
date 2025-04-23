
# import gym
# from gym import spaces
# import numpy as np
# import asyncio
# import websockets
# import json
# import threading
# from stable_baselines3 import PPO

# class DroneSimEnv(gym.Env):
#     def __init__(self):
#         super().__init__()

#         self.uri = "ws://localhost:8765"
#         self.websocket = None
#         self.loop = asyncio.new_event_loop()
#         self.thread = threading.Thread(target=self._start_loop, daemon=True)
#         self.thread.start()

#         self.action_space = spaces.MultiDiscrete([6, 201, 2])  # Speed 0–5, Altitude -100–100, Movement fwd/rev
#         self.observation_space = spaces.Box(
#             low=np.array([0, -500, 0, 0]), 
#             high=np.array([5, 1000, 100, 2]), 
#             dtype=np.float32
#         )

#         self.latest_telemetry = None
#         self.latest_metrics = None

#     def _start_loop(self):
#         asyncio.set_event_loop(self.loop)
#         self.loop.run_until_complete(self._connect())

#     async def _connect(self):
#         self.websocket = await websockets.connect(self.uri)
#         response = await self.websocket.recv()
#         data = json.loads(response)
#         print(f"Connected to server. ID: {data.get('connection_id')}")

#     def reset(self):
#         # Can't really reset the server, so we simulate it with new metrics
#         self.latest_telemetry = None
#         self.latest_metrics = None
#         return self._get_obs()

#     def _get_obs(self):
#         if not self.latest_telemetry:
#             return np.array([0, 0, 100, 2], dtype=np.float32)
#         try:
#             parts = self.latest_telemetry.split("-")
#             altitude = int(parts[3])
#             battery = float(parts[4].replace('%',''))
#             status = parts[-1].strip().upper()
#             sensor_status = {"RED": 0, "YELLOW": 1, "GREEN": 2}.get(status, 2)
#             speed = 0  # You can also keep track of speed manually

#             return np.array([speed, altitude, battery, sensor_status], dtype=np.float32)
#         except:
#             return np.array([0, 0, 100, 2], dtype=np.float32)

#     def step(self, action):
#         speed, alt_idx, movement = action
#         altitude = alt_idx - 100
#         movement = "fwd" if movement == 0 else "rev"

#         response = self.loop.run_until_complete(
#             self._send_command(speed, altitude, movement)
#         )

#         done = False
#         reward = 0
#         if response is None or response.get("status") == "crashed":
#             done = True
#             reward = -100
#         else:
#             self.latest_telemetry = response.get("telemetry", "")
#             self.latest_metrics = response.get("metrics", {})
#             distance = int(self.latest_metrics.get("total_distance", 0))
#             reward = 1 + distance / 100

#         return self._get_obs(), reward, done, {}

#     async def _send_command(self, speed, altitude, movement):
#         try:
#             command = {"speed": speed, "altitude": altitude, "movement": movement}
#             await self.websocket.send(json.dumps(command))
#             response = await self.websocket.recv()
#             return json.loads(response)
#         except:
#             return None

#     def close(self):
#         self.loop.run_until_complete(self.websocket.close())


# env = DroneSimEnv()
# model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=100_000)
# model.save("drone_autopilot_rl")



import gym
from gym import spaces
import numpy as np
import asyncio
import websockets
import json
import threading
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class DroneSimEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.uri = "ws://localhost:8765"
        self.websocket = None
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

        # Action space: speed (0–5), altitude (-100 to 100), direction (fwd/rev)
        self.action_space = spaces.MultiDiscrete([6, 201, 2])

        # Observation space (normalized): speed, altitude, battery, sensor_status
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]), 
            high=np.array([1.0, 1.0, 1.0, 1.0]), 
            dtype=np.float32
        )

        self.latest_telemetry = None
        self.latest_metrics = None
        self.current_speed = 0
        self._connecting = False  # Initialize the connecting flag

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect())

    async def _connect(self):
        if self._connecting:
            print("Already attempting to connect...")
            return

        self._connecting = True
        try:
            print("Attempting WebSocket connection...")
            self.websocket = await websockets.connect(self.uri)
            response = await self.websocket.recv()
            data = json.loads(response)
            print(f"Connected to server. ID: {data.get('connection_id')}")
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            # Retry connection after a short delay
            await asyncio.sleep(1)
            await self._connect()  # Retry the connection
        finally:
            self._connecting = False

    def _normalize_obs(self, speed, altitude, battery, sensor_status):
        return np.array([
            speed / 5.0,
            (altitude + 500) / 1500.0,  # Normalize -500 to 1000
            battery / 100.0,
            sensor_status / 2.0  # RED=0, GREEN=2
        ], dtype=np.float32)

    def _get_obs(self):
        if not self.latest_telemetry:
            return self._normalize_obs(0, 0, 100, 2)

        try:
            parts = self.latest_telemetry.split("-")
            altitude = int(parts[3])
            battery = float(parts[4].replace('%', ''))
            status = parts[-1].strip().upper()
            sensor_status = {"RED": 0, "YELLOW": 1, "GREEN": 2}.get(status, 2)

            return self._normalize_obs(self.current_speed, altitude, battery, sensor_status)
        except Exception as e:
            print("Obs parse error:", e)
            return self._normalize_obs(0, 0, 100, 2)

    def reset(self):
        self.latest_telemetry = None
        self.latest_metrics = None
        self.current_speed = 0
        return self._get_obs()

    def step(self, action):
        if self.websocket is None or not self.websocket.open:
            print("WebSocket is not connected or closed. Attempting to reconnect...")
            if not self._connecting:
                asyncio.run_coroutine_threadsafe(self._connect(), self.loop)

        speed, alt_idx, movement = action
        self.current_speed = speed
        altitude = alt_idx - 100
        movement_str = "fwd" if movement == 0 else "rev"

        # Send command asynchronously via the background loop
        future = asyncio.run_coroutine_threadsafe(
            self._send_command(speed, altitude, movement_str), self.loop
        )
        try:
            response = future.result(timeout=2)
        except Exception as e:
            print("WebSocket command error:", e)
            response = None

        reward = 0
        done = False

        if response is None or response.get("status") == "crashed":
            reward = -100
            done = True
        else:
            self.latest_telemetry = response.get("telemetry", "")
            self.latest_metrics = response.get("metrics", {})

            try:
                telemetry = self.latest_telemetry
                metrics = self.latest_metrics or {}

                distance = int(metrics.get("total_distance", 0))
                reward += distance / 100.0

                parts = telemetry.split("-")
                altitude = int(parts[3])
                battery = float(parts[4].replace('%', ''))
                status = parts[-1].strip().upper()

                # Battery penalty
                reward -= (100 - battery) / 100.0

                # Altitude limit enforcement
                if status == "RED" and altitude > 3:
                    reward -= 50
                    done = True
                elif status == "YELLOW" and altitude > 1000:
                    reward -= 20
                elif battery < 1:
                    reward -= 100
                    done = True

                # Small bonus for surviving
                reward += 1
            except Exception as e:
                print("Step parse error:", e)
                reward = -100
                done = True

        return self._get_obs(), reward, done, {}

    async def _send_command(self, speed, altitude, movement):
        try:
            if self.websocket is None or not self.websocket.open:
                print("WebSocket is closed. Reconnecting...")
                await self._connect()  # Reconnect

            command = {"speed": speed, "altitude": altitude, "movement": movement}
            await self.websocket.send(json.dumps(command))
            response = await self.websocket.recv()
            return json.loads(response)
        except Exception as e:
            print(f"WebSocket send/receive failed: {e}")
            return None

    def close(self):
        if self.websocket:
            asyncio.run_coroutine_threadsafe(self.websocket.close(), self.loop)


# === RL Training Script ===

if __name__ == "__main__":
    env = DummyVecEnv([lambda: DroneSimEnv()])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/")
    model.learn(total_timesteps=1_000_000)
    model.save("drone_autopilot_rl")

    print("✅ Training complete. Model saved as 'drone_autopilot_rl'")





