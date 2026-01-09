"""Quick test script for chat API"""
import requests
import time

print("Testing chat API...")
start = time.time()

try:
    response = requests.post(
        'http://localhost:8000/api/chat',
        json={'message': 'Hello'},
        timeout=60
    )
    elapsed = time.time() - start
    
    print(f"Response time: {elapsed:.2f} seconds")
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Response field exists: {'response' in data}")
        print(f"Response length: {len(data.get('response', ''))} characters")
        print(f"Response preview: {data.get('response', '')[:200]}")
        print("✅ API is working!")
    else:
        print(f"❌ Error: {response.status_code}")
        print(f"Response: {response.text}")
        
except requests.exceptions.Timeout:
    print("❌ Request timed out (took longer than 60 seconds)")
except requests.exceptions.ConnectionError as e:
    print(f"❌ Connection error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")


