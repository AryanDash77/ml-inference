import asyncio
import aiohttp
import time
import json
import statistics

TARGET_URL = "http://127.0.0.1:8000/predict"

# Test parameters
NUM_REQUESTS = 1000
CONCURRENT_USERS = 50

PAYLOAD = {
    "v": [0.0] * 28,  
    "amount": 150.50,
    "time": 3600.0
}

async def send_request(session, url, payload):
    start_time = time.time()
    try:
        async with session.post(url, json=payload, timeout=5) as response:
            await response.read()
            return time.time() - start_time, response.status
    except Exception as e:
        return time.time() - start_time, 500

async def main():
    print(f"Starting load test: {NUM_REQUESTS} requests with {CONCURRENT_USERS} concurrent users...")
    
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        all_results = []  # Initialize an empty list to store ALL results
        
        for _ in range(NUM_REQUESTS):
            tasks.append(send_request(session, TARGET_URL, PAYLOAD))
            
            # Throttle slightly to maintain steady concurrent load
            if len(tasks) >= CONCURRENT_USERS:
                batch_results = await asyncio.gather(*tasks)
                all_results.extend(batch_results)  # Add this batch to the total list
                tasks = []
        
        # Process any remaining tasks
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)

    # Calculate metrics using the full list
    latencies = [res[0] for res in all_results if res[1] == 200]
    success_rate = (len(latencies) / NUM_REQUESTS) * 100

    if latencies:
        p95 = statistics.quantiles(latencies, n=100)[94]
        avg = sum(latencies) / len(latencies)
        print("\n--- Load Test Results ---")
        print(f"Success Rate:   {success_rate:.1f}%")
        print(f"Average Latency: {avg * 1000:.2f} ms")
        print(f"p95 Latency:    {p95 * 1000:.2f} ms")
    else:
        print("\nAll requests failed. Check if the server is running and reachable.")

if __name__ == "__main__":
    asyncio.run(main())
           