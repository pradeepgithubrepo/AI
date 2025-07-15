import threading
import random
import time
import json
import os
import asyncio
from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData
from dotenv import load_dotenv
load_dotenv(override=True)

CONNECTION_STRING = os.getenv("EVENTHUB_CONNECTION_STRING")
ENTITY_NAME = os.getenv("EVENTHUB_ENTITY_NAME")
NUM_TRUCKS = int(os.getenv("NUM_TRUCKS", "100"))
POD_NAME = os.getenv("POD_NAME", "unknown-pod")
TRUCK_ID_OFFSET = int(os.getenv("TRUCK_ID_OFFSET", "0"))

# Realistic simulation parameters
START_LAT = 37.7749
START_LON = -122.4194
CRUISING_SPEED = 60  # km/h
FUEL_CONSUMPTION_PER_KM = 0.2  # liters/km
STOP_PROBABILITY = 0.01  # chance to stop each tick
STOP_DURATION_RANGE = (5, 20)  # seconds

class Truck:
    def __init__(self, truck_id):
        self.truck_id = truck_id
        self.latitude = START_LAT + random.uniform(-0.01, 0.01)
        self.longitude = START_LON + random.uniform(-0.01, 0.01)
        self.speed = CRUISING_SPEED + random.uniform(-5, 5)
        self.fuel_level = random.uniform(80, 100)
        self.temperature = random.uniform(15, 25)
        self.heading = random.uniform(0, 360)  # degrees
        self.stopped = False
        self.stop_time_left = 0

    def update(self):
        # Handle stoppages
        if self.stopped:
            self.speed = 0
            self.stop_time_left -= 1
            if self.stop_time_left <= 0:
                self.stopped = False
                self.speed = CRUISING_SPEED + random.uniform(-5, 5)
        else:
            # Randomly decide to stop
            if random.random() < STOP_PROBABILITY:
                self.stopped = True
                self.stop_time_left = random.randint(*STOP_DURATION_RANGE)
                self.speed = 0
            else:
                # Vary speed slightly
                self.speed = max(0, self.speed + random.uniform(-2, 2))
                # Move in a straight line based on heading and speed
                distance_km = self.speed / 3600  # per second
                # Approximate conversion for small distances
                delta_lat = distance_km / 111  # 1 deg lat ~ 111km
                delta_lon = distance_km / (111 * abs(math.cos(math.radians(self.latitude))) + 1e-6)
                self.latitude += delta_lat * math.cos(math.radians(self.heading))
                self.longitude += delta_lon * math.sin(math.radians(self.heading))
                # Fuel consumption
                self.fuel_level = max(0, self.fuel_level - distance_km * FUEL_CONSUMPTION_PER_KM)
        # Temperature drifts slowly
        self.temperature += random.uniform(-0.05, 0.05)

    def get_data(self):
        return {
            "pod": POD_NAME,
            "truck_id": self.truck_id,
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "speed": round(self.speed, 2),
            "fuel_level": round(self.fuel_level, 2),
            "temperature": round(self.temperature, 2),
            "timestamp": int(time.time())
        }

async def send_to_eventhub(data):
    producer = EventHubProducerClient.from_connection_string(
        conn_str=CONNECTION_STRING,
        eventhub_name=ENTITY_NAME
    )
    async with producer:
        event_data_batch = await producer.create_batch()
        event_data_batch.add(EventData(json.dumps(data)))
        await producer.send_batch(event_data_batch)

def simulate_truck(truck: Truck):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    while True:
        truck.update()
        data = truck.get_data()
        loop.run_until_complete(send_to_eventhub(data))
        print(f"[{POD_NAME}] Sent to Event Hub: {json.dumps(data)}")
        time.sleep(1)

def start_simulation(num_trucks=NUM_TRUCKS, truck_id_offset=TRUCK_ID_OFFSET):
    threads = []
    for i in range(num_trucks):
        truck_id = truck_id_offset + i + 1
        truck = Truck(truck_id=truck_id)
        t = threading.Thread(target=simulate_truck, args=(truck,), daemon=True)
        t.start()
        threads.append(t)
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Simulation stopped.")

if __name__ == "__main__":
    import math
    print(f"Starting simulation in pod {POD_NAME} with {NUM_TRUCKS} trucks. Truck IDs: {TRUCK_ID_OFFSET + 1} to {TRUCK_ID_OFFSET + NUM_TRUCKS}")
    start_simulation()
