"""
Kafka producer to stream hospital observations into the system.

Reads synthetic data and publishes to Kafka topics for real-time processing.
"""

import json
import os
from pathlib import Path
from typing import Any

from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()


class HospitalDataProducer:
    """Stream vitals, labs, and microbiology data to Kafka topics."""

    def __init__(self, bootstrap_servers: str | None = None):
        """
        Initialize the Kafka producer.
        
        Args:
            bootstrap_servers: Kafka broker address. Defaults to env var.
        """
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
        )
        
        self.config = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": "hospital_data_producer",
            "compression.type": "snappy",
            "batch.size": 16384,
            "linger.ms": 10,
        }
        
        self.producer = Producer(self.config)
        
        self.topic_vitals = os.getenv("KAFKA_TOPIC_VITALS", "hospital.vitals")
        self.topic_labs = os.getenv("KAFKA_TOPIC_LABS", "hospital.labs")
        self.topic_micro = os.getenv("KAFKA_TOPIC_MICROBIOLOGY", "hospital.microbiology")

    def delivery_callback(self, err: Any, msg: Any) -> None:
        """Callback to confirm message delivery."""
        if err:
            print(f"ERROR: Message delivery failed: {err}")
        else:
            print(f"✓ Delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}")

    def produce_vitals(self, vitals_file: Path) -> None:
        """Publish vitals to Kafka topic."""
        with open(vitals_file) as f:
            vitals = json.load(f)
        
        for record in vitals:
            key = record["patient_id"].encode("utf-8")
            value = json.dumps(record).encode("utf-8")
            self.producer.produce(
                topic=self.topic_vitals,
                key=key,
                value=value,
                callback=self.delivery_callback,
            )
        
        self.producer.flush()
        print(f"Produced {len(vitals)} vitals records")

    def produce_labs(self, labs_file: Path) -> None:
        """Publish lab results to Kafka topic."""
        with open(labs_file) as f:
            labs = json.load(f)
        
        for record in labs:
            key = record["patient_id"].encode("utf-8")
            value = json.dumps(record).encode("utf-8")
            self.producer.produce(
                topic=self.topic_labs,
                key=key,
                value=value,
                callback=self.delivery_callback,
            )
        
        self.producer.flush()
        print(f"Produced {len(labs)} labs records")

    def produce_microbiology(self, cultures_file: Path) -> None:
        """Publish culture results to Kafka topic."""
        with open(cultures_file) as f:
            cultures = json.load(f)
        
        for record in cultures:
            key = record["patient_id"].encode("utf-8")
            value = json.dumps(record).encode("utf-8")
            self.producer.produce(
                topic=self.topic_micro,
                key=key,
                value=value,
                callback=self.delivery_callback,
            )
        
        self.producer.flush()
        print(f"Produced {len(cultures)} culture records")

    def produce_site_data(self, site_dir: Path) -> None:
        """Produce all data for one hospital site."""
        print(f"Streaming data from {site_dir}...")
        
        vitals_file = site_dir / "vitals.json"
        labs_file = site_dir / "labs.json"
        cultures_file = site_dir / "cultures.json"
        
        if vitals_file.exists():
            self.produce_vitals(vitals_file)
        
        if labs_file.exists():
            self.produce_labs(labs_file)
        
        if cultures_file.exists():
            self.produce_microbiology(cultures_file)

    def close(self) -> None:
        """Flush and close the producer."""
        self.producer.flush()


def main() -> None:
    """Stream synthetic data from all three sites to Kafka."""
    producer = HospitalDataProducer()
    
    data_root = Path("data/synthetic")
    
    for site in ["site_a", "site_b", "site_c"]:
        site_dir = data_root / site
        if site_dir.exists():
            producer.produce_site_data(site_dir)
    
    producer.close()
    print("\n✓ All data streamed to Kafka")


if __name__ == "__main__":
    main()

