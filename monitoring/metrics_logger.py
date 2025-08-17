"""
metrics_logger.py - Module for logging metrics to InfluxDB for Grafana visualization
"""

import os
import time
import socket
from datetime import datetime
import logging
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import psutil
import numpy as np
from typing import Dict, Any, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("metrics_logger")

class MetricsLogger:
    """
    Class to log metrics from cyberbullying detection models to InfluxDB
    for visualization in Grafana.
    """
    
    def __init__(
        self,
        url: str = "http://localhost:8086",
        token: str = "",  # Set your InfluxDB token here or use environment variable
        org: str = "cyberbullying_detection",
        bucket: str = "model_metrics",
        batch_size: int = 10,
        enabled: bool = True,
        debug: bool = False
    ):
        """Initialize the metrics logger"""
        # Set debug mode if requested
        if debug:
            logger.setLevel(logging.DEBUG)
            
        self.enabled = enabled
        if not enabled:
            logger.info("Metrics logging is disabled")
            return
            
        logger.debug(f"Initializing metrics logger with URL: {url}, Token: {'*' * (len(token) if token else 0)}")
        
        self.url = url
        self.token = token or os.getenv("INFLUXDB_TOKEN", "")
        self.org = org
        self.bucket = bucket
        self.batch_size = batch_size
        self.hostname = socket.gethostname()
        
        # Print diagnostic info
        logger.debug(f"Hostname: {self.hostname}")
        logger.debug(f"Organization: {self.org}")
        logger.debug(f"Bucket: {self.bucket}")
        
        try:
            logger.debug(f"Attempting to connect to InfluxDB at {self.url} with token length: {len(self.token)}")
            self.client = InfluxDBClient(url=self.url, token=self.token, org=self.org)
            
            # Test the connection
            health = self.client.health()
            logger.debug(f"InfluxDB health check: {health}")
            
            # Check if bucket exists
            buckets_api = self.client.buckets_api()
            buckets = buckets_api.find_buckets().buckets
            bucket_names = [bucket.name for bucket in buckets]
            logger.debug(f"Available buckets: {bucket_names}")
            
            if self.bucket not in bucket_names:
                logger.warning(f"Bucket '{self.bucket}' not found! Available buckets: {bucket_names}")
                logger.warning("Creating bucket...")
                try:
                    buckets_api.create_bucket(bucket_name=self.bucket, org=self.org)
                    logger.info(f"Created bucket '{self.bucket}'")
                except Exception as e:
                    logger.error(f"Failed to create bucket: {str(e)}")
            
            self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            logger.info(f"✅ Successfully connected to InfluxDB at {self.url}")
            
            # Write a test point to verify everything works
            try:
                test_point = Point("connection_test").tag("host", self.hostname).field("connected", 1)
                self.write_api.write(bucket=self.bucket, record=test_point)
                logger.debug("Successfully wrote test point")
            except Exception as e:
                logger.error(f"Could write test point, but connection established: {str(e)}")
                
        except Exception as e:
            logger.error(f"Failed to connect to InfluxDB: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.enabled = False
    
    def log_prediction(
        self,
        model_name: str,
        text_id: str,
        prediction: int,
        confidence: float,
        ground_truth: int = None,
        latency_ms: float = None,
        text_length: int = None,
        additional_metrics: Dict[str, Any] = None
    ) -> None:
        """
        Log a single prediction event with relevant metrics
        
        Args:
            model_name: Name of the model (bert, bilstm, emotion_fusion, logistic)
            text_id: Identifier for the text being classified
            prediction: Predicted class (0 or 1)
            confidence: Prediction confidence score
            ground_truth: Actual class if available
            latency_ms: Prediction latency in milliseconds
            text_length: Length of the input text
            additional_metrics: Any additional metrics to log
        """
        if not self.enabled:
            logger.debug("Metrics logging is disabled, not logging prediction")
            return
        
        try:
            logger.debug(f"Creating point for prediction: model={model_name}, prediction={prediction}, confidence={confidence:.4f}")
            
            point = Point("model_prediction") \
                .tag("model", model_name) \
                .tag("hostname", self.hostname) \
                .tag("text_id", text_id) \
                .field("prediction", prediction) \
                .field("confidence", float(confidence))
                
            if text_length is not None:
                point = point.field("text_length", int(text_length))
                
            if ground_truth is not None:
                point = point.field("ground_truth", int(ground_truth))
                point = point.field("correct", int(prediction == ground_truth))
            
            if latency_ms is not None:
                point = point.field("latency_ms", float(latency_ms))
                
            if additional_metrics:
                for key, value in additional_metrics.items():
                    if isinstance(value, (int, float, bool, str)):
                        point = point.field(key, value)
            
            logger.debug(f"Writing point to bucket '{self.bucket}'")
            self.write_api.write(bucket=self.bucket, record=point)
            logger.debug(f"✅ Successfully logged prediction metrics for model {model_name}")
        except Exception as e:
            logger.error(f"Failed to log prediction metrics: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def log_system_metrics(self) -> None:
        """Log system metrics like CPU, memory usage"""
        if not self.enabled:
            return
            
        try:
            cpu_percent = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            
            point = Point("system_metrics") \
                .tag("hostname", self.hostname) \
                .field("cpu_percent", cpu_percent) \
                .field("memory_percent", mem.percent) \
                .field("memory_used_mb", mem.used / (1024 * 1024))
                
            self.write_api.write(bucket=self.bucket, record=point)
            logger.debug("Logged system metrics")
        except Exception as e:
            logger.error(f"Failed to log system metrics: {str(e)}")
    
    def log_batch_metrics(
        self,
        model_name: str,
        predictions: List[int],
        confidence_scores: List[float],
        ground_truths: List[int] = None,
        latencies_ms: List[float] = None
    ) -> None:
        """
        Log metrics for a batch of predictions
        
        Args:
            model_name: Name of the model
            predictions: List of predictions (0 or 1)
            confidence_scores: List of confidence scores
            ground_truths: List of actual classes if available
            latencies_ms: List of prediction latencies
        """
        if not self.enabled:
            return
            
        try:
            if ground_truths:
                accuracy = sum(p == gt for p, gt in zip(predictions, ground_truths)) / len(predictions)
                avg_confidence = np.mean(confidence_scores)
                
                point = Point("batch_metrics") \
                    .tag("model", model_name) \
                    .tag("hostname", self.hostname) \
                    .field("accuracy", accuracy) \
                    .field("avg_confidence", avg_confidence) \
                    .field("batch_size", len(predictions))
                
                if latencies_ms:
                    point = point.field("avg_latency_ms", np.mean(latencies_ms))
                    point = point.field("max_latency_ms", np.max(latencies_ms))
                
                self.write_api.write(bucket=self.bucket, record=point)
                logger.info(f"Logged batch metrics for model {model_name}: accuracy={accuracy:.4f}")
        except Exception as e:
            logger.error(f"Failed to log batch metrics: {str(e)}")
    
    def close(self) -> None:
        """Close the InfluxDB client connection"""
        if self.enabled:
            try:
                self.client.close()
                logger.info("Closed connection to InfluxDB")
            except Exception as e:
                logger.error(f"Error closing InfluxDB connection: {str(e)}")
