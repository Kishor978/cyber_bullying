# Grafana Monitoring Integration Guide

This guide explains how to integrate the Grafana monitoring setup with the Cyberbullying Detection System.

## Detailed Setup Steps

### Step 1: Install Docker

Since we'll use Docker for easier deployment, ensure Docker is installed:

1. If you don't have Docker installed:
   - For Windows: Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - For Linux: Run `sudo apt-get install docker.io docker-compose` (Ubuntu) or equivalent for your distribution

2. Verify Docker is running:
   ```bash
   docker --version
   ```

### Step 2: Prepare the Monitoring Infrastructure

1. Navigate to your project directory:
   ```bash
   cd e:\Cyberbullying
   ```

2. Make sure the monitoring directory structure exists:
   ```bash
   mkdir -p monitoring/grafana/provisioning/datasources
   ```

3. Start InfluxDB and Grafana using Docker Compose:
   ```bash
   cd monitoring
   docker-compose up -d
   ```
   
   This command will pull the necessary Docker images and start both InfluxDB and Grafana containers in the background.

4. Wait for services to start (usually about 10-30 seconds)

### Step 3: Access Grafana

Open your browser and go to http://localhost:3000. Log in with:
- Username: `admin`
- Password: `admin`

You'll be prompted to change the password on first login.

### Step 4: Configure InfluxDB Data Source

If the data source wasn't automatically configured:

1. In the Grafana UI, click on the gear icon (⚙️) in the left sidebar to open Settings
2. Select "Data sources"
3. Click "Add data source"
4. Select "InfluxDB"
5. Configure the data source with:
   - Name: `InfluxDB`
   - URL: `http://influxdb:8086` (this is the Docker service name)
   - Database: Select "Flux" as the query language
   - Organization: `cyberbullying_detection`
   - Token: `cyberbullying_token` (this should match what's in docker-compose.yml)
   - Default bucket: `model_metrics`
6. Click "Save & Test" to verify the connection

### Step 5: Import the Dashboard

1. In the Grafana UI, hover over the "+" icon in the sidebar and click "Import"
2. Click "Upload JSON file" and select `monitoring/grafana_dashboard.json`
3. Select the InfluxDB data source in the dropdown
4. Click "Import"

### Step 6: Verify Data is Flowing

Now you need to run your application to generate some metrics:

1. Open a new terminal window
2. Run your Streamlit application:
   ```bash
   cd e:\Cyberbullying
   streamlit run deployment/app.py
   ```

3. Use the application to make a few predictions:
   - Select different models
   - Enter some text for classification
   - This will generate metrics data for Grafana

4. Return to the Grafana dashboard and wait for metrics to appear
   - You should start seeing data in the dashboard panels after making predictions
   - It may take a few minutes for initial data to appear

### Step 7: Explore the Dashboard

The dashboard provides several panels:
- Model accuracy over time
- Average prediction confidence
- Cyberbullying detection distribution
- Prediction latency by model
- System resource utilization

You can:
- Adjust the time range in the top-right corner
- Hover over graphs to see detailed data
- Click and drag to zoom into specific time periods

### Step 8: Customizing the Dashboard

If you want to customize the dashboard:
1. Click the settings icon (gear) at the top of the dashboard
2. Select "Edit"
3. You can now:
   - Move panels by dragging them
   - Resize panels by dragging their borders
   - Edit panels by clicking on their titles and selecting "Edit"
4. Save your changes with the disk icon at the top right

## Metrics Being Tracked

The monitoring setup tracks the following metrics:

1. **Model Performance**
   - Prediction accuracy
   - Confidence scores
   - Prediction latency

2. **System Metrics**
   - CPU usage
   - Memory usage

3. **Usage Analytics**
   - Number of predictions
   - Distribution of cyberbullying vs non-cyberbullying predictions

## Extending the Metrics

To add new metrics:

1. Add fields to the appropriate method in `metrics_logger.py`
2. Update the Grafana dashboard to display the new metrics

## Troubleshooting

### Data Not Appearing in Grafana

If your containers are running but data isn't showing up in Grafana dashboards:

1. **Check Container Status:**
   ```bash
   docker ps
   ```
   Ensure both InfluxDB and Grafana containers are running and healthy.

2. **Check InfluxDB Logs:**
   ```bash
   docker logs $(docker ps -q --filter "name=influxdb")
   ```
   Look for any error messages or connection issues.

3. **Verify Data is Being Written:**
   Access the InfluxDB UI at http://localhost:8086 and login with:
   - Username: `admin`
   - Password: `cyberbullying_admin`
   
   Navigate to "Data Explorer" and check if data is being written to the `model_metrics` bucket.

4. **Test Direct Connection to InfluxDB:**
   Run this simple Python script to test InfluxDB connectivity:
   ```python
   from influxdb_client import InfluxDBClient, Point
   from influxdb_client.client.write_api import SYNCHRONOUS
   
   client = InfluxDBClient(
       url="http://localhost:8086",
       token="cyberbullying_token",
       org="cyberbullying_detection"
   )
   
   write_api = client.write_api(write_options=SYNCHRONOUS)
   
   # Create a test point
   p = Point("test_measurement").tag("test_tag", "test_value").field("test_field", 123.0)
   
   # Write the point
   write_api.write(bucket="model_metrics", record=p)
   print("Test point written successfully!")
   client.close()
   ```

5. Replace all occurrences of ${DS_INFLUXDB} with the actual UID of your InfluxDB data source.

To find your InfluxDB data source UID:

Go to Grafana -> Configuration -> Data Sources
Click on your InfluxDB data source
Look at the URL - the last part is the UID (e.g., ```http://localhost:3000/datasources/edit/abc123``` - "abc123" is the UID)

6. **Check Network Connectivity:**
   - If running in Docker, ensure the containers can communicate with each other
   - If running locally, verify no firewall is blocking port 8086

7. **Verify Time Range in Grafana:**
   - Check that your Grafana dashboard time range (top-right corner) includes recent data
   - Try setting to "Last 5 minutes" or "Last 15 minutes"

8. **Restart Services:**
   ```bash
   cd monitoring
   docker-compose down
   docker-compose up -d
   ```

9. **Check Python Dependencies:**
   Ensure `influxdb-client` is properly installed:
   ```bash
   pip install --upgrade influxdb-client
   ```

10. **Verify Configuration Values:**
    Double-check that organization name, bucket name, and token values match between:
    - Your Python code
    - The docker-compose.yml file
    - The Grafana data source configuration
