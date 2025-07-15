import os
import asyncio
from azure.eventhub.aio import EventHubProducerClient
from azure.eventhub import EventData
from datetime import datetime, timezone
import random

CONNECTION_STRING = "Endpoint=sb://esehdbzthsp5821llfnsyx.servicebus.windows.net/;SharedAccessKeyName=key_d079d2fe-35f7-414b-b8e1-c69120777933;SharedAccessKey=zk/ZVn5nFdJcCDfV4XLOtpQJkQP5p1mi4+AEhLcG6f8="
ENTITY_NAME = "es_167bc92a-e453-4cb9-8f14-76d281796cde"

def get_row_data(id):
    time = datetime.now(timezone.utc).isoformat()
    deviceID = id + 100
    humidity = random.randint(35, 65)
    temperature = random.randint(20, 37)
    return {
        "entryTime": time,
        "messageId": id,
        "temperature": temperature,
        "humidity": humidity,
        "deviceID": deviceID
    }

async def main():
    producer = EventHubProducerClient.from_connection_string(
        conn_str=CONNECTION_STRING,
        eventhub_name=ENTITY_NAME
    )

    batch_size = 10
    batch_count = 5

    async with producer:
        for j in range(batch_count):
            event_data_batch = await producer.create_batch()
            for k in range(batch_size):
                event_data_batch.add(EventData(str(get_row_data(k))))
            await producer.send_batch(event_data_batch)
            print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
                  f"[Send events to Fabric Eventstream]: batch#{j} ({batch_size} events) has been sent to eventstream")
            await asyncio.sleep(1)
    print(datetime.now().strftime('%Y/%m/%d %H:%M:%S'),
          f"[Send events to Fabric Eventstream]: All {batch_count} batches have been sent to eventstream")

if __name__ == "__main__":
    asyncio.run(main())