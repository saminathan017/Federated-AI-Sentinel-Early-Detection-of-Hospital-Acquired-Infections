"""
Kafka consumer to receive hospital observations from the event stream.

Consumes vitals, labs, and microbiology data for feature extraction and model training.
"""

import json
import os
from typing import Any, Callable

from confluent_kafka import Consumer, KafkaError, KafkaException
from dotenv import load_dotenv

load_dotenv()


class HospitalDataConsumer:
    """Consume vitals, labs, and microbiology data from Kafka topics."""

    def __init__(
        self,
        group_id: str = "sentinel_consumer",
        bootstrap_servers: str | None = None,
    ):
        """
        Initialize the Kafka consumer.
        
        Args:
            group_id: Consumer group ID for offset management
            bootstrap_servers: Kafka broker address. Defaults to env var.
        """
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        
        self.config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": True,
            "session.timeout.ms": 6000,
        }
        
        self.consumer = Consumer(self.config)
        
        self.topic_vitals = os.getenv("KAFKA_TOPIC_VITALS", "hospital.vitals")
        self.topic_labs = os.getenv("KAFKA_TOPIC_LABS", "hospital.labs")
        self.topic_micro = os.getenv("KAFKA_TOPIC_MICROBIOLOGY", "hospital.microbiology")

    def subscribe_all(self) -> None:
        """Subscribe to all hospital data topics."""
        topics = [self.topic_vitals, self.topic_labs, self.topic_micro]
        self.consumer.subscribe(topics)
        print(f"Subscribed to topics: {topics}")

    def consume_batch(
        self,
        batch_size: int = 100,
        timeout: float = 5.0,
    ) -> list[dict[str, Any]]:
        """
        Consume a batch of messages.
        
        Args:
            batch_size: Maximum number of messages to consume
            timeout: Timeout in seconds per poll
        
        Returns:
            List of decoded message payloads
        """
        messages = []
        
        for _ in range(batch_size):
            msg = self.consumer.poll(timeout)
            
            if msg is None:
                break
            
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    # End of partition
                    continue
                else:
                    raise KafkaException(msg.error())
            
            try:
                value = json.loads(msg.value().decode("utf-8"))
                messages.append({
                    "topic": msg.topic(),
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "key": msg.key().decode("utf-8") if msg.key() else None,
                    "data": value,
                })
            except json.JSONDecodeError as e:
                print(f"ERROR: Failed to decode message: {e}")
        
        return messages

    def consume_stream(
        self,
        callback: Callable[[dict[str, Any]], None],
        max_messages: int | None = None,
    ) -> None:
        """
        Consume messages in streaming mode and pass each to a callback.
        
        Args:
            callback: Function to call with each decoded message
            max_messages: Stop after this many messages, or run indefinitely if None
        """
        count = 0
        
        try:
            while True:
                msg = self.consumer.poll(timeout=1.0)
                
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        raise KafkaException(msg.error())
                
                try:
                    value = json.loads(msg.value().decode("utf-8"))
                    message_dict = {
                        "topic": msg.topic(),
                        "partition": msg.partition(),
                        "offset": msg.offset(),
                        "key": msg.key().decode("utf-8") if msg.key() else None,
                        "data": value,
                    }
                    callback(message_dict)
                    count += 1
                    
                    if max_messages and count >= max_messages:
                        break
                
                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to decode message: {e}")
        
        except KeyboardInterrupt:
            print("\nStopping consumer...")
        
        finally:
            self.close()

    def close(self) -> None:
        """Close the consumer and commit offsets."""
        self.consumer.close()
        print("Consumer closed")


def example_callback(message: dict[str, Any]) -> None:
    """Example callback that prints each consumed message."""
    print(f"Received from {message['topic']}: {message['data']['patient_id']}")


def main() -> None:
    """Example: consume messages in batch mode."""
    consumer = HospitalDataConsumer(group_id="example_consumer")
    consumer.subscribe_all()
    
    print("Consuming messages (Ctrl+C to stop)...")
    
    try:
        while True:
            batch = consumer.consume_batch(batch_size=10, timeout=2.0)
            if batch:
                print(f"Received {len(batch)} messages")
                for msg in batch:
                    print(f"  {msg['topic']}: {msg['data']['patient_id']}")
            else:
                print("No messages, waiting...")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        consumer.close()


if __name__ == "__main__":
    main()

