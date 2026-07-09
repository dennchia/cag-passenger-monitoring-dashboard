1. Update Backend Dependencies
cd "C:\Users\aveng\Documents\Codex\CAG (MP)\backend"
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

2. Start MQTT Broker
& "C:\Program Files\mosquitto\mosquitto.exe" -c "C:\Users\aveng\Documents\Codex\CAG (MP)\mosquitto_server.conf" -v

3. Start Full Server Mode
cd "C:\Users\aveng\Documents\Codex\CAG (MP)"
.\start_server.ps1

4. Quick Checks
Invoke-RestMethod "http://localhost:8000/health"
Test-NetConnection YOUR_LAPTOP_IP -Port 8000
Test-NetConnection YOUR_LAPTOP_IP -Port 1883


5. Strong pc should use this connection
--mqtt-broker 192.168.50.197 `
--mqtt-port 1883